"""
Module pour les recommandations hybrides combinant plusieurs approches
"""
import logging
import sys
from collections import defaultdict

# Import configuration
sys.path.append('..')
from config import ES_INDEX_TRACKS, RECOMMENDATION_WEIGHTS
from utils.elasticsearch_client import get_elasticsearch_client

# Import des algorithmes de recommandation
from recommendation_algorithms.content_based import content_based_recommend, get_track_features
from recommendation_algorithms.collaborative import collaborative_recommend, get_user_listening_history
from recommendation_algorithms.mood_based import mood_based_recommend, get_user_mood
from recommendation_algorithms.popularity import get_popular_songs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_user_tags(user_id, es=None):
    """
    Obtient les tags préférés d'un utilisateur basés sur son historique d'écoute
    
    Args:
        user_id (str): ID de l'utilisateur
        es: Client Elasticsearch (si None, un nouveau client sera créé)
        
    Returns:
        list: Liste des tags les plus courants
    """
    # Obtenir le client Elasticsearch si non fourni
    if es is None:
        es = get_elasticsearch_client()
        if es is None:
            logger.error("Client Elasticsearch non disponible")
            return []
    
    # Obtenir l'historique d'écoute de l'utilisateur
    history = get_user_listening_history(user_id, limit=20, es=es)
    
    if not history:
        logger.warning(f"Aucun historique d'écoute trouvé pour l'utilisateur {user_id}")
        return []
    
    # Obtenir les IDs des pistes écoutées
    track_ids = [item['track_id'] for item in history]
    
    # Obtenir les détails des pistes
    tracks_query = {
        "query": {
            "terms": {
                "id": track_ids
            }
        },
        "size": len(track_ids)
    }
    
    try:
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Compter les occurrences de tags
        tag_counts = defaultdict(int)
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            if 'tags' in track and track['tags']:
                for tag in track['tags']:
                    tag_counts[tag] += 1
        
        # Trier les tags par nombre d'occurrences
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Retourner les 10 tags les plus courants
        return [tag for tag, count in sorted_tags[:10]]
    except Exception as e:
        logger.error(f"Erreur lors de l'obtention des tags utilisateur: {e}")
        return []

def hybrid_recommend(user_id=None, seed_tracks=None, n=10, es=None):
    """
    Recommandation hybride combinant plusieurs approches
    
    Args:
        user_id (str, optional): ID de l'utilisateur
        seed_tracks (list, optional): Liste d'IDs de pistes de référence
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
    
    # Poids par défaut pour les différents types de recommandation
    weights = RECOMMENDATION_WEIGHTS
    
    # Initialiser les scores
    track_scores = defaultdict(float)
    
    # Recommandations basées sur le contenu à partir des pistes de référence ou des tags utilisateur
    if seed_tracks:
        for track_id in seed_tracks:
            content_recs = content_based_recommend(track_id=track_id, n=20, es=es)
            for rec in content_recs:
                track_scores[rec['id']] += weights['content'] * rec['similarity_score']
    elif user_id:
        # Obtenir les tags utilisateur à partir de l'historique d'écoute
        user_tags = get_user_tags(user_id, es=es)
        if user_tags:
            content_recs = content_based_recommend(user_tags=user_tags, n=20, es=es)
            for rec in content_recs:
                track_scores[rec['id']] += weights['content'] * rec['similarity_score']
    
    # Recommandations par filtrage collaboratif
    if user_id:
        collab_recs = collaborative_recommend(user_id, n=20, es=es)
        for rec in collab_recs:
            track_scores[rec['id']] += weights['collaborative'] * rec['score']
        
        # Recommandations basées sur l'humeur
        current_mood = get_user_mood(user_id, es=es)
        if current_mood:
            mood_recs = mood_based_recommend(current_mood, n=20, es=es)
            for rec in mood_recs:
                track_scores[rec['id']] += weights['mood'] * rec['score']
    
    # Recommandations par popularité (comme fallback)
    pop_recs = get_popular_songs(n=20, es=es)
    for rec in pop_recs:
        track_scores[rec['id']] += weights['popularity'] * (rec['popularity'] / 100 if 'popularity' in rec else 0.5)
    
    # Si nous n'avons toujours pas assez de recommandations, ajouter plus de chansons populaires
    if len(track_scores) < n:
        more_pop_recs = get_popular_songs(n=n*2, es=es)
        for rec in more_pop_recs:
            if rec['id'] not in track_scores:
                track_scores[rec['id']] = weights['popularity'] * (rec['popularity'] / 100 if 'popularity' in rec else 0.5)
    
    # Trier et obtenir les N meilleures
    top_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Obtenir les détails des pistes
    track_ids = [track_id for track_id, _ in top_tracks]
    
    if not track_ids:
        logger.warning("Aucune piste recommandée trouvée")
        return []
    
    tracks_query = {
        "query": {
            "terms": {
                "id": track_ids
            }
        },
        "size": n
    }
    
    try:
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Créer une correspondance entre ID de piste et détails de piste
        tracks_map = {hit['_source']['id']: hit['_source'] for hit in tracks_results['hits']['hits']}
        
        # Formater les recommandations
        recommendations = []
        for track_id, score in top_tracks:
            if track_id in tracks_map:
                track = tracks_map[track_id]
                recommendations.append({
                    'id': track_id,
                    'name': track.get('name', 'Unknown'),
                    'artists': track.get('artist_names', 'Unknown'),
                    'score': float(score)
                })
        
        logger.info(f"Recommandation hybride: {len(recommendations)} pistes trouvées")
        return recommendations
    except Exception as e:
        logger.error(f"Erreur dans la recommandation hybride: {e}")
        return []

if __name__ == "__main__":
    # Test du module
    es = get_elasticsearch_client()
    
    # Test avec un ID utilisateur
    user_id = "user_42"  # Remplacer par un ID valide
    recommendations = hybrid_recommend(user_id=user_id, n=5, es=es)
    print(f"Recommandations hybrides pour l'utilisateur {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['score']:.4f})")
    
    # Test avec des pistes de référence
    seed_tracks = ["0UaMYEvWZi0ZqiDOoHU3YI"]  # Remplacer par des IDs valides
    recommendations = hybrid_recommend(seed_tracks=seed_tracks, n=5, es=es)
    print(f"\nRecommandations hybrides basées sur les pistes {seed_tracks}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} par {rec['artists']} (score: {rec['score']:.4f})")