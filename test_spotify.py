import os
import logging
import requests
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from config import SPOTIFY_CONFIG
from kafka import KafkaProducer
import json
from spotify_sync import SpotifySync
from kafka_consumer import RecommendationConsumer
from recommendation.hybrid_recommender import HybridRecommender
from recommendation.sentiment_analyzer import SentimentAnalyzer

# Charger les variables d'environnement si besoin
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_spotify_connection():
    """Test de la connexion à l'API Spotify"""
    try:
        # Initialiser le client Spotify avec OAuth
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CONFIG['client_id'],
            client_secret=SPOTIFY_CONFIG['client_secret'],
            redirect_uri=SPOTIFY_CONFIG['redirect_uri'],
            scope=SPOTIFY_CONFIG['scope'],
            open_browser=True
        ))

        # Test 1: Recherche de chansons
        logger.info("Test 1: Recherche de chansons...")
        results = sp.search(q='artist:Ed Sheeran', limit=5, type='track')
        for track in results['tracks']['items']:
            logger.info(f"Trouvé: {track['name']} par {track['artists'][0]['name']}")

        # Test 2: Détails d'une chanson
        if results['tracks']['items']:
            track_id = results['tracks']['items'][0]['id']
            logger.info(f"\nTest 2: Détails de la chanson {track_id}...")
            try:
                track_details = sp.track(track_id)
                logger.info(f"Titre: {track_details['name']}")
                logger.info(f"Artiste: {track_details['artists'][0]['name']}")
                logger.info(f"Popularité: {track_details['popularity']}")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des détails: {str(e)}")

        # Test 3: Caractéristiques audio
        if results['tracks']['items']:
            logger.info("\nTest 3: Caractéristiques audio...")
            try:
                features = sp.audio_features([track_id])[0]
                if features:
                    logger.info(f"Danceability: {features['danceability']}")
                    logger.info(f"Energy: {features['energy']}")
                    logger.info(f"Tempo: {features['tempo']}")
                    logger.info(f"Valence: {features['valence']}")
                    logger.info(f"Acousticness: {features['acousticness']}")
                else:
                    logger.error("Aucune caractéristique audio trouvée")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des caractéristiques audio: {str(e)}")
                logger.error("Vérifiez que vous avez les bonnes permissions dans votre application Spotify")

        logger.info("\nTous les tests ont réussi!")
        return True

    except Exception as e:
        logger.error(f"Erreur lors du test: {str(e)}")
        return False


class SpotifyKafkaSync:
    def __init__(self):
        # Configuration Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CONFIG['client_id'],
            client_secret=SPOTIFY_CONFIG['client_secret'],
            redirect_uri=SPOTIFY_CONFIG['redirect_uri'],
            scope=SPOTIFY_CONFIG['scope']
        ))

        # Configuration Kafka
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    def sync_user_history(self, user_id):
        """Synchronise l'historique utilisateur vers Kafka"""
        try:
            # Récupérer l'historique Spotify
            history = self.sp.current_user_recently_played(limit=50)

            # Envoyer à Kafka
            for item in history['items']:
                event = {
                    'user_id': user_id,
                    'track_id': item['track']['id'],
                    'played_at': item['played_at'],
                    'event_type': 'play'
                }
                self.producer.send('user_events', value=event)

            return True
        except Exception as e:
            print(f"Erreur: {str(e)}")
            return False


def main():
    # Initialiser la synchronisation
    sync = SpotifyKafkaSync()

    # Synchroniser les données
    user_id = "votre_user_id"
    sync.sync_user_history(user_id)

    # Initialiser le consommateur
    consumer = RecommendationConsumer()

    # Démarrer le traitement
    consumer.process_events()


if __name__ == "__main__":
    main()
