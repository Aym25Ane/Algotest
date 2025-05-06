"""
Module pour les recommandations basées sur la popularité
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

def get_popular_songs(n=10, es=None):
    """
    Obtient les chansons populaires basées sur les données Elasticsearch
    
    Args:
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
    
    # Utiliser le champ popularity si disponible, sinon utiliser le nombre d'écoutes des interactions
    query = {
        "query": {
            "match_all": {}
        },
        "sort": [
            {"popularity": {"order": "desc"}}
        ],
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'popularity': track.get('popularity', 0)
            })
        
        logger.info(f"Recommandation par popularité: {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation par popularité: {e}")
        return []

def get_trending_songs(time_range="week", n=10, es=None):
    """
    Obtient les chansons tendance basées sur les interactions récentes
    
    Args:
        time_range (str): Plage de temps pour les tendances ("day", "week", "month")
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
    
    # Déterminer la plage de temps
    time_ranges = {
        "day": "now-1d/d",
        "week": "now-1w/d",
        "month": "now-1M/d"
    }
    
    range_value = time_ranges.get(time_range, "now-1w/d")
    
    # Requête pour obtenir les pistes les plus écoutées récemment
    query = {
        "query": {
            "range": {
                "timestamp": {
                    "gte": range_value
                }
            }
        },
        "aggs": {
            "trending_tracks": {
                "terms": {
                    "field": "track_id",
                    "size": n * 2,
                    "order": {"play_count_sum": "desc"}
                },
                "aggs": {
                    "play_count_sum": {
                        "sum": {
                            "field": "play_count"
                        }
                    }
                }
            }
        },
        "size": 0
    }
    
    try:
        # Cette requête nécessite l'indice user_interactions
        # Si cet indice n'existe pas ou est vide, utiliser get_popular_songs à la place
        if not es.indices.exists(index="spotify_user_interactions"):
            logger.warning("L'indice spotify_user_interactions n'existe pas, utilisation de get_popular_songs à la place")
            return get_popular_songs(n=n, es=es)
        
        results = es.search(index="spotify_user_interactions", body=query)
        
        # Extraire les IDs de pistes tendance
        trending_track_ids = [
            bucket['key'] 
            for bucket in results['aggregations']['trending_tracks']['buckets']
        ]
        
        if not trending_track_ids:
            logger.warning("Aucune piste tendance trouvée, utilisation de get_popular_songs à la place")
            return get_popular_songs(n=n, es=es)
        
        # Obtenir les détails des pistes
        tracks_query = {
            "query": {
                "terms": {
                    "id": trending_track_ids
                }
            },
            "size": n
        }
        
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Formater les recommandations
        recommendations = []
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'trending_score': hit['_score']
            })
        
        logger.info(f"Recommandation par tendance ({time_range}): {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation par tendance: {e}")
        # En cas d'erreur, utiliser get_popular_songs comme fallback
        return get_popular_songs(n=n, es=es)

def get_genre_popular_songs(genre, n=10, es=None):
    """
    Obtient les chansons populaires d'un genre spécifique
    
    Args:
        genre (str): Genre musical
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
    
    query = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"track_genre": genre}},
                    {"term": {"tags": genre}}
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [
            {"popularity": {"order": "desc"}}
        ],
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'popularity': track.get('popularity', 0),
                'genre': genre
            })
        
        logger.info(f"Recommandation par popularité pour le genre '{genre}': {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation par popularité pour le genre '{genre}': {e}")
        return []

def get_new_releases(days=30, n=10, es=None):
    """
    Obtient les nouvelles sorties
    
    Args:
        days (int): Nombre de jours à considérer comme "nouveau"
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
    
    # Calculer la date limite
    from datetime import datetime, timedelta
    date_limit = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "release_date": {
                                "gte": date_limit
                            }
                        }
                    }
                ]
            }
        },
        "sort": [
            {"release_date": {"order": "desc"}},
            {"popularity": {"order": "desc"}}
        ],
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'release_date': track.get('release_date', 'Unknown'),
                'popularity': track.get('popularity', 0)
            })
        
        logger.info(f"Recommandation de nouvelles sorties (derniers {days} jours): {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation de nouvelles sorties: {e}")
        return []

if __name__ == "__main__":
    # Test du module
    es = get_elasticsearch_client()
    
    # Test pour obtenir les chansons populaires
    recommendations = get_popular_songs(n=5, es=es)
    print("Recommandations par popularité:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (popularité: {rec['popularity']})")
    
    # Test pour obtenir les chansons tendance
    trending_recommendations = get_trending_songs(time_range="week", n=5, es=es)
    print("\nRecommandations par tendance (semaine):")
    for i, rec in enumerate(trending_recommendations, 1):
        score = rec.get('trending_score', 0)
        print(f"{i}. {rec['name']} par {rec['artists']} (score de tendance: {score:.2f})")
    
    # Test pour obtenir les chansons populaires d'un genre
    genre = "pop"  # Remplacer par un genre valide
    genre_recommendations = get_genre_popular_songs(genre=genre, n=5, es=es)
    print(f"\nRecommandations populaires du genre '{genre}':")
    for i, rec in enumerate(genre_recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (popularité: {rec['popularity']})")
    
    # Test pour obtenir les nouvelles sorties
    new_releases = get_new_releases(days=90, n=5, es=es)
    print("\nNouvelles sorties (90 derniers jours):")
    for i, rec in enumerate(new_releases, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (date de sortie: {rec['release_date']})")