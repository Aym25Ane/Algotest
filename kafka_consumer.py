from kafka import KafkaConsumer
import json
import logging
from elasticsearch import Elasticsearch
from recommendation.hybrid_recommender import HybridRecommender
from spotify_sync import SpotifySync
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from kafka import KafkaProducer
from config import SPOTIFY_CONFIG, ES_CONFIG , KAFKA_CONFIG
from recommendation.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationConsumer:
    def __init__(self, kafka_bootstrap_servers, es_client, spotify_sync):
        self.consumer = KafkaConsumer(
            'user-interactions',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='recommendation-group'
        )
        self.es = es_client
        self.spotify_sync = spotify_sync
        self.recommender = HybridRecommender(es_client)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_event(self, event):
        """Process a single user interaction event"""
        try:
            event_type = event.get('type')
            song_id = event.get('songId')
            user_id = event.get('userId')
            timestamp = event.get('timestamp')

            if not all([event_type, song_id, user_id, timestamp]):
                logger.error(f"Invalid event format: {event}")
                return

            # Ensure track exists in our database
            await self.spotify_sync.sync_track(song_id)

            # Process different event types
            metrics = {
                'viewCount': 0,
                'reactionCount': 0,
                'commentCount': 0,
                'comment': None
            }

            if event_type == 'LISTEN':
                metrics['viewCount'] = 1
                # Update user listening history for collaborative filtering
                await self.update_user_history(user_id, song_id, timestamp)
            elif event_type == 'LIKE':
                metrics['reactionCount'] = 1
            elif event_type == 'COMMENT':
                metrics['commentCount'] = 1
                metrics['comment'] = {
                    'userId': user_id,
                    'text': event.get('commentText', ''),
                    'timestamp': timestamp
                }

            # Update track metrics
            await self.spotify_sync.update_track_metrics(song_id, metrics)

            # Update recommendations
            await self.update_recommendations(user_id)

        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")

    async def update_user_history(self, user_id: str, song_id: str, timestamp: str):
        """Update user listening history for collaborative filtering"""
        try:
            self.es.index(
                index='user_history',
                body={
                    'userId': user_id,
                    'songId': song_id,
                    'timestamp': timestamp,
                    'type': 'LISTEN'
                }
            )
        except Exception as e:
            logger.error(f"Error updating user history: {str(e)}")

    async def update_recommendations(self, user_id):
        """Update recommendations for a user"""
        try:
            # Get hybrid recommendations
            recommendations = self.recommender.get_recommendations(user_id)

            # Store recommendations with explanation
            for rec in recommendations:
                explanation = self.recommender.get_recommendation_explanation(
                    user_id,
                    rec['id']
                )
                rec['explanation'] = explanation

            # Store in Elasticsearch
            self.es.index(
                index='user_recommendations',
                id=user_id,
                body={
                    'userId': user_id,
                    'recommendations': recommendations,
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error updating recommendations: {str(e)}")

    async def start(self):
        """Start consuming messages"""
        logger.info("Starting recommendation consumer...")
        for message in self.consumer:
            try:
                event = message.value
                # Process event in a separate thread to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    lambda: asyncio.run(self.process_event(event))
                )
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")


class SpotifyDataCollector:
    def __init__(self):
        # Initialiser Spotify
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CONFIG['client_id'],
            client_secret=SPOTIFY_CONFIG['client_secret'],
            redirect_uri=SPOTIFY_CONFIG['redirect_uri'],
            scope=SPOTIFY_CONFIG['scope']
        ))

        # Initialiser Elasticsearch
        self.es = Elasticsearch([ELASTICSEARCH_CONFIG['host']])

    def fetch_and_index_song(self, track_id):
        """Récupère et indexe les données d'une chanson"""
        try:
            # Récupérer les métadonnées
            track = self.sp.track(track_id)

            # Récupérer les caractéristiques audio
            features = self.sp.audio_features([track_id])[0]

            # Préparer le document
            document = {
                'track_id': track_id,
                'name': track['name'],
                'artist': track['artists'][0]['name'],
                'album': track['album']['name'],
                'popularity': track['popularity'],
                'features': features
            }

            # Indexer dans Elasticsearch
            self.es.index(
                index='songs',
                id=track_id,
                body=document
            )

            return True
        except Exception as e:
            print(f"Erreur: {str(e)}")
            return False


if __name__ == "__main__":
    # Initialize clients
    es_client = Elasticsearch(['http://localhost:9200'])
    spotify_sync = SpotifySync(
        client_id='your_client_id',
        client_secret='your_client_secret',
        es_client=es_client
    )

    # Create and start consumer
    consumer = RecommendationConsumer(
        kafka_bootstrap_servers=['localhost:9092'],
        es_client=es_client,
        spotify_sync=spotify_sync
    )

    # Run the consumer
    asyncio.run(consumer.start())