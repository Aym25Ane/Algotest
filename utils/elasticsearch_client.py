"""
Client Elasticsearch pour le système de recommandation
"""
import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError
import sys
import time

# Import configuration
sys.path.append('..')
from config import ES_HOST, ES_PORT, ES_SCHEME, ES_USER, ES_PASSWORD
from config import ES_INDEX_TRACKS, ES_INDEX_ARTISTS, ES_INDEX_USER_INTERACTIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_elasticsearch_client():
    """
    Crée et retourne un client Elasticsearch
    Avec gestion des tentatives de connexion
    """
    # Configuration de connexion
    es_config = {
        'host': ES_HOST,
        'port': ES_PORT,
        'scheme': ES_SCHEME
    }
    
    # Ajouter l'authentification si nécessaire
    if ES_USER and ES_PASSWORD:
        es_config['http_auth'] = (ES_USER, ES_PASSWORD)
    
    # Créer le client avec retry_on_timeout
    es = Elasticsearch([es_config], retry_on_timeout=True, max_retries=5)
    
    # Vérifier la connexion avec des tentatives
    max_attempts = 5
    attempt = 0
    connected = False
    
    while attempt < max_attempts and not connected:
        try:
            if es.ping():
                logger.info("Connecté à Elasticsearch")
                connected = True
            else:
                attempt += 1
                logger.warning(f"Échec de connexion à Elasticsearch (tentative {attempt}/{max_attempts})")
                time.sleep(5)  # Attendre 5 secondes avant de réessayer
        except ConnectionError as e:
            attempt += 1
            logger.warning(f"Erreur de connexion à Elasticsearch: {e} (tentative {attempt}/{max_attempts})")
            time.sleep(5)
    
    if not connected:
        logger.error("Impossible de se connecter à Elasticsearch après plusieurs tentatives")
        return None
    
    return es

def create_indices(es):
    """
    Crée les indices Elasticsearch nécessaires s'ils n'existent pas déjà
    """
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return False
    
    # Tracks index mapping
    tracks_mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                "popularity": {"type": "float"},
                "duration_ms": {"type": "integer"},
                "explicit": {"type": "boolean"},
                "artists": {"type": "keyword"},
                "artist_names": {"type": "text", "analyzer": "standard"},
                "id_artists": {"type": "keyword"},
                "release_date": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy"},
                "danceability": {"type": "float"},
                "energy": {"type": "float"},
                "key": {"type": "integer"},
                "loudness": {"type": "float"},
                "mode": {"type": "integer"},
                "speechiness": {"type": "float"},
                "acousticness": {"type": "float"},
                "instrumentalness": {"type": "float"},
                "liveness": {"type": "float"},
                "valence": {"type": "float"},
                "tempo": {"type": "float"},
                "time_signature": {"type": "integer"},
                "track_genre": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "mood": {"type": "keyword"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # Artists index mapping
    artists_mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                "genres": {"type": "keyword"},
                "popularity": {"type": "float"},
                "followers": {"type": "integer"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # User interactions index mapping
    interactions_mapping = {
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "track_id": {"type": "keyword"},
                "play_count": {"type": "integer"},
                "timestamp": {"type": "date"},
                "liked": {"type": "boolean"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # Create indices if they don't exist
    indices_created = 0
    
    if not es.indices.exists(index=ES_INDEX_TRACKS):
        es.indices.create(index=ES_INDEX_TRACKS, body=tracks_mapping)
        logger.info(f"Indice créé: {ES_INDEX_TRACKS}")
        indices_created += 1
    else:
        logger.info(f"L'indice {ES_INDEX_TRACKS} existe déjà")
    
    if not es.indices.exists(index=ES_INDEX_ARTISTS):
        es.indices.create(index=ES_INDEX_ARTISTS, body=artists_mapping)
        logger.info(f"Indice créé: {ES_INDEX_ARTISTS}")
        indices_created += 1
    else:
        logger.info(f"L'indice {ES_INDEX_ARTISTS} existe déjà")
    
    if not es.indices.exists(index=ES_INDEX_USER_INTERACTIONS):
        es.indices.create(index=ES_INDEX_USER_INTERACTIONS, body=interactions_mapping)
        logger.info(f"Indice créé: {ES_INDEX_USER_INTERACTIONS}")
        indices_created += 1
    else:
        logger.info(f"L'indice {ES_INDEX_USER_INTERACTIONS} existe déjà")
    
    logger.info(f"{indices_created} indices créés, {3 - indices_created} indices existaient déjà")
    return True

def delete_indices(es, confirm=False):
    """
    Supprime les indices Elasticsearch (à utiliser avec précaution)
    """
    if not confirm:
        logger.warning("Suppression des indices annulée. Passez confirm=True pour confirmer la suppression.")
        return False
    
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return False
    
    indices_deleted = 0
    
    if es.indices.exists(index=ES_INDEX_TRACKS):
        es.indices.delete(index=ES_INDEX_TRACKS)
        logger.info(f"Indice supprimé: {ES_INDEX_TRACKS}")
        indices_deleted += 1
    
    if es.indices.exists(index=ES_INDEX_ARTISTS):
        es.indices.delete(index=ES_INDEX_ARTISTS)
        logger.info(f"Indice supprimé: {ES_INDEX_ARTISTS}")
        indices_deleted += 1
    
    if es.indices.exists(index=ES_INDEX_USER_INTERACTIONS):
        es.indices.delete(index=ES_INDEX_USER_INTERACTIONS)
        logger.info(f"Indice supprimé: {ES_INDEX_USER_INTERACTIONS}")
        indices_deleted += 1
    
    logger.info(f"{indices_deleted} indices supprimés")
    return True

def get_index_stats(es):
    """
    Retourne des statistiques sur les indices
    """
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return None
    
    stats = {}
    
    for index in [ES_INDEX_TRACKS, ES_INDEX_ARTISTS, ES_INDEX_USER_INTERACTIONS]:
        if es.indices.exists(index=index):
            count = es.count(index=index)['count']
            stats[index] = {
                'document_count': count,
                'size': es.indices.stats(index=index)['indices'][index]['total']['store']['size_in_bytes']
            }
        else:
            stats[index] = {
                'document_count': 0,
                'size': 0
            }
    
    return stats