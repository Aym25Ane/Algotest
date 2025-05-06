"""
Configuration globale pour le système de recommandation Spotify
"""
import os

# Elasticsearch configuration
ES_HOST = os.environ.get("ES_HOST", "localhost")
ES_PORT = int(os.environ.get("ES_PORT", 9200))
ES_SCHEME = os.environ.get("ES_SCHEME", "http")
ES_USER = os.environ.get("ES_USER", "")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")

# Elasticsearch indices
ES_INDEX_TRACKS = "spotify_tracks"
ES_INDEX_ARTISTS = "spotify_artists"
ES_INDEX_USER_INTERACTIONS = "spotify_user_interactions"

# Kaggle dataset
KAGGLE_DATASET = "maharshipandya/-spotify-tracks-dataset"
DATASET_PATH = os.environ.get("DATASET_PATH", "./data")

# API configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", 5000))
API_DEBUG = os.environ.get("API_DEBUG", "True").lower() == "true"

# Recommendation weights
RECOMMENDATION_WEIGHTS = {
    'content': float(os.environ.get("WEIGHT_CONTENT", 0.4)),
    'collaborative': float(os.environ.get("WEIGHT_COLLABORATIVE", 0.3)),
    'mood': float(os.environ.get("WEIGHT_MOOD", 0.2)),
    'popularity': float(os.environ.get("WEIGHT_POPULARITY", 0.1))
}

# Audio features for content-based recommendation
AUDIO_FEATURES = [
    'acousticness', 
    'danceability', 
    'energy', 
    'instrumentalness', 
    'liveness', 
    'loudness', 
    'speechiness', 
    'tempo', 
    'valence'
]

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'