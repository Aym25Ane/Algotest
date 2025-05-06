"""
Service API Flask pour les recommandations musicales
"""
import logging
import sys
from flask import Flask, request, jsonify, current_app
import pandas as pd

# Import configuration
sys.path.append('..')
from config import API_HOST, API_PORT, API_DEBUG
from utils.elasticsearch_client import get_elasticsearch_client

# Import des algorithmes de recommandation
from recommendation_algorithms.content_based import content_based_recommend
from recommendation_algorithms.collaborative import collaborative_recommend
from recommendation_algorithms.mood_based import mood_based_recommend, get_user_mood
from recommendation_algorithms.popularity import get_popular_songs
from recommendation_algorithms.hybrid import hybrid_recommend

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Créer l'application Flask
app = Flask(__name__)

# Client Elasticsearch global
es = None

# Fonction d'initialisation
def initialize_elasticsearch():
    """Initialise le client Elasticsearch"""
    global es
    es = get_elasticsearch_client()
    if es is None:
        logger.error("Impossible de se connecter à Elasticsearch")
    else:
        logger.info("Connecté à Elasticsearch")

# Initialiser Elasticsearch au démarrage
with app.app_context():
    initialize_elasticsearch()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    global es
    es_status = "healthy" if es and es.ping() else "unhealthy"
    return jsonify({
        'status': 'healthy',
        'elasticsearch': es_status
    })

@app.route('/recommendations/content-based', methods=['GET'])
def get_content_based():
    """Obtient des recommandations basées sur le contenu"""
    track_id = request.args.get('track_id')
    user_tags = request.args.get('user_tags')
    n = int(request.args.get('n', 10))
    
    if user_tags:
        user_tags = user_tags.split(',')
    
    if not track_id and not user_tags:
        return jsonify({'error': 'Le paramètre track_id ou user_tags est requis'}), 400
    
    recommendations = content_based_recommend(track_id=track_id, user_tags=user_tags, n=n, es=es)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/collaborative', methods=['GET'])
def get_collaborative():
    """Obtient des recommandations par filtrage collaboratif"""
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 10))
    
    if not user_id:
        return jsonify({'error': 'Le paramètre user_id est requis'}), 400
    
    recommendations = collaborative_recommend(user_id, n, es=es)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/mood', methods=['GET'])
def get_mood_based():
    """Obtient des recommandations basées sur l'humeur"""
    mood = request.args.get('mood')
    n = int(request.args.get('n', 10))
    
    if not mood:
        return jsonify({'error': 'Le paramètre mood est requis'}), 400
    
    recommendations = mood_based_recommend(mood, n, es=es)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/popular', methods=['GET'])
def get_popular():
    """Obtient des recommandations basées sur la popularité"""
    n = int(request.args.get('n', 10))
    recommendations = get_popular_songs(n, es=es)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/hybrid', methods=['GET'])
def get_hybrid():
    """Obtient des recommandations hybrides"""
    user_id = request.args.get('user_id')
    seed_tracks = request.args.get('seed_tracks')
    n = int(request.args.get('n', 10))
    
    # Analyser les pistes de référence si fournies
    seed_track_list = None
    if seed_tracks:
        seed_track_list = seed_tracks.split(',')
    
    recommendations = hybrid_recommend(user_id, seed_track_list, n, es=es)
    return jsonify({'recommendations': recommendations})

@app.route('/user/<user_id>/mood', methods=['GET'])
def get_user_current_mood(user_id):
    """Obtient l'humeur actuelle d'un utilisateur basée sur les écoutes récentes"""
    mood = get_user_mood(user_id, es=es)
    return jsonify({'mood': mood})

@app.route('/track-interaction', methods=['POST'])
def track_interaction():
    """Enregistre l'interaction d'un utilisateur avec une piste"""
    data = request.json
    
    if not data or 'user_id' not in data or 'track_id' not in data or 'interaction_type' not in data:
        return jsonify({'error': 'Champs requis manquants'}), 400
    
    user_id = data['user_id']
    track_id = data['track_id']
    interaction_type = data['interaction_type']
    
    # Créer le document d'interaction
    interaction = {
        'user_id': user_id,
        'track_id': track_id,
        'timestamp': pd.Timestamp.now().isoformat(),
        'play_count': 1 if interaction_type == 'play' else 0,
        'liked': interaction_type == 'like'
    }
    
    # Générer un ID unique pour l'interaction
    interaction_id = f"{user_id}_{track_id}_{pd.Timestamp.now().timestamp()}"
    
    try:
        # Indexer l'interaction
        es.index(index="spotify_user_interactions", id=interaction_id, body=interaction)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de l'interaction: {e}")
        return jsonify({'error': str(e)}), 500

def run_api_server():
    """Démarre le serveur API Flask"""
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)

if __name__ == "__main__":
    run_api_server()