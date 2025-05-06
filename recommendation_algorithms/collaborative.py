"""
Module pour les recommandations par filtrage collaboratif
"""
import logging
import sys

# Import configuration
sys.path.append('..')
from config import ES_INDEX_TRACKS, ES_INDEX_USER_INTERACTIONS
from utils.elasticsearch_client import get_elasticsearch_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collaborative_recommend(user_id, n=10, es=None):
    """
    Recommandation par filtrage collaboratif utilisant Elasticsearch
    
    Args:
        user_id (str): ID de l'utilisateur
        n (int): Nombre de recommandations à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Liste de recommandations
    """
    if not user_id:
        logger.error("Aucun user_id fourni pour la recommandation collaborative")
        return []
    
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    # Étape 1: Obtenir l'historique d'écoute de l'utilisateur
    user_query = {
        "query": {
            "term": {"user_id": user_id}
        },
        "size": 100  # Limiter aux 100 interactions récentes
    }
    
    try:
        user_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=user_query)
        
        if len(user_results['hits']['hits']) == 0:
            logger.warning(f"Aucun historique d'écoute trouvé pour l'utilisateur {user_id}")
            return []
        
        # Extraire les pistes que l'utilisateur a écoutées
        user_tracks = [hit['_source']['track_id'] for hit in user_results['hits']['hits']]
        
        # Étape 2: Trouver des utilisateurs avec un historique d'écoute similaire
        similar_users_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"track_id": user_tracks}},
                        {"bool": {"must_not": {"term": {"user_id": user_id}}}}
                    ]
                }
            },
            "aggs": {
                "similar_users": {
                    "terms": {
                        "field": "user_id",
                        "size": 20,
                        "order": {"track_overlap": "desc"}
                    },
                    "aggs": {
                        "track_overlap": {
                            "cardinality": {
                                "field": "track_id"
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        similar_users_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=similar_users_query)
        
        similar_users = [
            bucket['key'] 
            for bucket in similar_users_results['aggregations']['similar_users']['buckets']
        ]
        
        if not similar_users:
            logger.warning(f"Aucun utilisateur similaire trouvé pour l'utilisateur {user_id}")
            return []
        
        # Étape 3: Obtenir les pistes que les utilisateurs similaires ont écoutées mais pas l'utilisateur
        recommendations_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"user_id": similar_users}}
                    ],
                    "must_not": [
                        {"terms": {"track_id": user_tracks}}
                    ]
                }
            },
            "aggs": {
                "track_recommendations": {
                    "terms": {
                        "field": "track_id",
                        "size": n * 2,  # Obtenir plus que nécessaire pour filtrer plus tard
                        "order": {"avg_play_count": "desc"}
                    },
                    "aggs": {
                        "avg_play_count": {
                            "avg": {
                                "field": "play_count"
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        recommendations_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=recommendations_query)
        
        # Extraire les IDs de pistes recommandées et les scores
        track_scores = {
            bucket['key']: bucket['avg_play_count']['value']
            for bucket in recommendations_results['aggregations']['track_recommendations']['buckets']
        }
        
        # Étape 4: Obtenir les détails des pistes
        if not track_scores:
            logger.warning("Aucune piste recommandée trouvée")
            return []
        
        tracks_query = {
            "query": {
                "terms": {
                    "id": list(track_scores.keys())
                }
            },
            "size": n
        }
        
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Formater les recommandations
        recommendations = []
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            track_id = track['id']
            recommendations.append({
                'id': track_id,
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'score': track_scores[track_id]
            })
        
        # Trier par score et limiter à n
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:n]
        
        logger.info(f"Recommandation collaborative: {len(recommendations)} pistes trouvées pour l'utilisateur {user_id}")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation collaborative: {e}")
        return []

def get_user_listening_history(user_id, limit=50, es=None):
    """
    Obtient l'historique d'écoute d'un utilisateur
    
    Args:
        user_id (str): ID de l'utilisateur
        limit (int): Nombre maximum d'entrées à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Historique d'écoute de l'utilisateur
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    query = {
        "query": {
            "term": {"user_id": user_id}
        },
        "sort": [
            {"timestamp": {"order": "desc"}}
        ],
        "size": limit
    }
    
    try:
        results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=query)
        
        history = []
        for hit in results['hits']['hits']:
            interaction = hit['_source']
            history.append(interaction)
        
        return history
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention de l'historique d'écoute de l'utilisateur {user_id}: {e}")
        return []

def find_similar_users(user_id, max_users=20, es=None):
    """
    Trouve des utilisateurs similaires basés sur l'historique d'écoute
    
    Args:
        user_id (str): ID de l'utilisateur
        max_users (int): Nombre maximum d'utilisateurs similaires à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Liste d'utilisateurs similaires avec leurs scores de similarité
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    # Obtenir l'historique d'écoute de l'utilisateur
    user_history = get_user_listening_history(user_id, limit=100, es=es)
    
    if not user_history:
        logger.warning(f"Aucun historique d'écoute trouvé pour l'utilisateur {user_id}")
        return []
    
    # Extraire les IDs de pistes
    track_ids = [interaction['track_id'] for interaction in user_history]
    
    # Trouver des utilisateurs qui ont écouté les mêmes pistes
    query = {
        "query": {
            "bool": {
                "must": [
                    {"terms": {"track_id": track_ids}},
                    {"bool": {"must_not": {"term": {"user_id": user_id}}}}
                ]
            }
        },
        "aggs": {
            "similar_users": {
                "terms": {
                    "field": "user_id",
                    "size": max_users,
                    "order": {"track_overlap": "desc"}
                },
                "aggs": {
                    "track_overlap": {
                        "cardinality": {
                            "field": "track_id"
                        }
                    }
                }
            }
        },
        "size": 0
    }
    
    try:
        results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=query)
        
        similar_users = []
        for bucket in results['aggregations']['similar_users']['buckets']:
            similar_users.append({
                'user_id': bucket['key'],
                'overlap_score': bucket['track_overlap']['value']
            })
        
        return similar_users
    except Exception as e:
        logger.error(f"Erreur lors de la recherche d'utilisateurs similaires: {e}")
        return []

if __name__ == "__main__":
    # Test du module
    es = get_elasticsearch_client()
    
    # Test avec un ID utilisateur
    user_id = "user_42"  # Remplacer par un ID valide
    recommendations = collaborative_recommend(user_id=user_id, n=5, es=es)
    print(f"Recommandations collaboratives pour l'utilisateur {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['score']:.2f})")
    
    # Test pour trouver des utilisateurs similaires
    similar_users = find_similar_users(user_id=user_id, max_users=5, es=es)
    print(f"\nUtilisateurs similaires à {user_id}:")
    for i, user in enumerate(similar_users, 1):
        print(f"{i}. {user['user_id']} (score de chevauchement: {user['overlap_score']:.2f})")