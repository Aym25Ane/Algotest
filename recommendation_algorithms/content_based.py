"""
Module pour les recommandations basées sur le contenu
"""
import logging
import sys

# Import configuration
sys.path.append('..')
from config import ES_INDEX_TRACKS
from utils.elasticsearch_client import get_elasticsearch_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def content_based_recommend(track_id=None, user_tags=None, n=10, es=None):
    """
    Recommandation basée sur le contenu utilisant Elasticsearch
    Peut recommander basé sur une piste de référence ou des tags utilisateur
    
    Args:
        track_id (str, optional): ID de la piste de référence
        user_tags (list, optional): Liste de tags utilisateur
        n (int): Nombre de recommandations à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Liste de recommandations
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    if track_id:
        # Obtenir les détails de la piste
        try:
            track = es.get(index=ES_INDEX_TRACKS, id=track_id)['_source']
        except Exception as e:
            logger.error(f"Piste {track_id} non trouvée: {e}")
            return []
        
        # Extraire les caractéristiques pour la recherche de similarité
        query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must_not": [
                                {"term": {"id": track_id}}  # Exclure la piste de référence
                            ],
                            "should": []
                        }
                    },
                    "functions": []
                }
            },
            "size": n
        }
        
        # Ajouter la correspondance d'humeur
        if 'mood' in track and track['mood']:
            query["query"]["function_score"]["query"]["bool"]["should"].append(
                {"term": {"mood": track['mood']}}
            )
        
        # Ajouter la correspondance de tags
        if 'tags' in track and track['tags']:
            query["query"]["function_score"]["query"]["bool"]["should"].append(
                {"terms": {"tags": track['tags']}}
            )
        
        # Ajouter la similarité des caractéristiques audio
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for feature in audio_features:
            if feature in track:
                query["query"]["function_score"]["functions"].append({
                    "gauss": {
                        feature: {
                            "origin": track[feature],
                            "scale": "0.2"
                        }
                    },
                    "weight": 1.0
                })
        
    elif user_tags:
        # Recommander basé sur les tags utilisateur
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"terms": {"tags": user_tags}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": n
        }
    else:
        logger.error("Aucun track_id ou user_tags fourni pour la recommandation basée sur le contenu")
        return []
    
    # Exécuter la requête
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'similarity_score': hit['_score']
            })
        
        logger.info(f"Recommandation basée sur le contenu: {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation basée sur le contenu: {e}")
        return []

def get_track_features(track_id, es=None):
    """
    Obtient les caractéristiques d'une piste
    
    Args:
        track_id (str): ID de la piste
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        dict: Caractéristiques de la piste
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return None
    
    try:
        track = es.get(index=ES_INDEX_TRACKS, id=track_id)['_source']
        return track
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des caractéristiques de la piste {track_id}: {e}")
        return None

def get_similar_tracks_by_features(features, n=10, es=None):
    """
    Trouve des pistes similaires basées sur des caractéristiques spécifiques
    
    Args:
        features (dict): Dictionnaire de caractéristiques à utiliser pour la recherche
        n (int): Nombre de recommandations à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Liste de recommandations
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    # Construire la requête
    query = {
        "query": {
            "function_score": {
                "query": {
                    "match_all": {}
                },
                "functions": []
            }
        },
        "size": n
    }
    
    # Ajouter les fonctions de score pour chaque caractéristique
    for feature, value in features.items():
        if feature in ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
            query["query"]["function_score"]["functions"].append({
                "gauss": {
                    feature: {
                        "origin": value,
                        "scale": "0.2"
                    }
                },
                "weight": 1.0
            })
    
    # Exécuter la requête
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'similarity_score': hit['_score']
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recherche de pistes similaires par caractéristiques: {e}")
        return []

if __name__ == "__main__":
    # Test du module
    es = get_elasticsearch_client()
    
    # Test avec un ID de piste
    track_id = "0UaMYEvWZi0ZqiDOoHU3YI"  # Remplacer par un ID valide
    recommendations = content_based_recommend(track_id=track_id, n=5, es=es)
    print(f"Recommandations basées sur la piste {track_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['similarity_score']:.2f})")
    
    # Test avec des tags utilisateur
    user_tags = ["energetic", "danceable", "electronic"]
    recommendations = content_based_recommend(user_tags=user_tags, n=5, es=es)
    print(f"\nRecommandations basées sur les tags {user_tags}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['similarity_score']:.2f})")