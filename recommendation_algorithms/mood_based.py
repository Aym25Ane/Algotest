"""
Module pour les recommandations basées sur l'humeur
"""
import logging
import sys
from collections import defaultdict

# Import configuration
sys.path.append('..')
from config import ES_INDEX_TRACKS, ES_INDEX_USER_INTERACTIONS
from utils.elasticsearch_client import get_elasticsearch_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mood_based_recommend(mood, n=10, es=None):
    """
    Recommandations basées sur l'humeur

    Args:
        mood (str): Humeur pour laquelle obtenir des recommandations
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
            "function_score": {
                "query": {
                    "term": {"mood": mood}
                },
                "random_score": {},  # Ajouter de l'aléatoire pour la variété
                "boost_mode": "sum"
            }
        },
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
                'score': hit['_score']
            })

        logger.info(f"Recommandation basée sur l'humeur '{mood}': {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation basée sur l'humeur: {e}")
        return []

def get_user_recent_listens(user_id, n=5, es=None):
    """
    Obtient les écoutes récentes d'un utilisateur

    Args:
        user_id (str): ID de l'utilisateur
        n (int): Nombre d'écoutes récentes à retourner
        es: Client Elasticsearch (si None, un nouveau client sera créé)

    Returns:
        list: Liste des pistes récemment écoutées
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
        "size": n
    }

    try:
        results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=query)
        recent_track_ids = [hit['_source']['track_id'] for hit in results['hits']['hits']]

        if not recent_track_ids:
            logger.warning(f"Aucune écoute récente trouvée pour l'utilisateur {user_id}")
            return []

        # Obtenir les détails des pistes
        tracks_query = {
            "query": {
                "terms": {
                    "id": recent_track_ids
                }
            },
            "size": n
        }

        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)

        recent_tracks = []
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            recent_tracks.append(track)

        return recent_tracks
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des écoutes récentes de l'utilisateur: {e}")
        return []

def extract_current_mood(tracks):
    """
    Extrait l'humeur actuelle à partir des pistes

    Args:
        tracks (list): Liste de pistes

    Returns:
        str: Humeur détectée
    """
    if not tracks:
        return "neutral"

    # Compter les occurrences d'humeur
    mood_counts = defaultdict(int)
    for track in tracks:
        if 'mood' in track:
            mood_counts[track['mood']] += 1

    # Retourner l'humeur la plus courante, ou neutre si aucune n'est trouvée
    if mood_counts:
        return max(mood_counts.items(), key=lambda x: x[1])[0]

    # Si aucune information d'humeur, essayer d'inférer à partir des caractéristiques audio
    valence_sum = sum(track.get('valence', 0.5) for track in tracks)
    energy_sum = sum(track.get('energy', 0.5) for track in tracks)

    avg_valence = valence_sum / len(tracks)
    avg_energy = energy_sum / len(tracks)

    # Déterminer l'humeur basée sur valence et énergie
    if avg_valence > 0.6 and avg_energy > 0.6:
        return "happy"
    elif avg_valence > 0.6 and avg_energy <= 0.4:
        return "relaxed"
    elif avg_valence <= 0.4 and avg_energy > 0.6:
        return "intense"
    elif avg_valence <= 0.4 and avg_energy <= 0.4:
        return "sad"
    else:
        return "neutral"

def get_user_mood(user_id, es=None):
    """
    Obtient l'humeur actuelle d'un utilisateur basée sur ses écoutes récentes

    Args:
        user_id (str): ID de l'utilisateur
        es: Client Elasticsearch (si None, un nouveau client sera créé)

    Returns:
        str: Humeur détectée
    """
    # Obtenir les écoutes récentes
    recent_tracks = get_user_recent_listens(user_id, n=10, es=es)

    # Extraire l'humeur
    mood = extract_current_mood(recent_tracks)

    logger.info(f"Humeur détectée pour l'utilisateur {user_id}: {mood}")
    return mood

if __name__ == "__main__":
    # Test du module
    es = get_elasticsearch_client()

    # Test avec une humeur
    mood = "happy"
    recommendations = mood_based_recommend(mood=mood, n=5, es=es)
    print(f"Recommandations basées sur l'humeur '{mood}':")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['score']:.2f})")

    # Test pour obtenir l'humeur d'un utilisateur
    user_id = "user_42"  # Remplacer par un ID valide
    user_mood = get_user_mood(user_id=user_id, es=es)
    print(f"\nHumeur détectée pour l'utilisateur {user_id}: {user_mood}")