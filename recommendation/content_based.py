from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get user data including profile and preferences."""
        try:
            user = self.es.get(index="users", id=user_id)
            return user["_source"]
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            return {}

    def _get_song_data(self, song_id: str) -> Dict[str, Any]:
        """Get song data including metadata."""
        try:
            song = self.es.get(index="songs", id=song_id)
            return song["_source"]
        except Exception as e:
            logger.error(f"Error fetching song data: {str(e)}")
            return {}

    def _calculate_song_similarity(self, song1: Dict[str, Any], song2: Dict[str, Any]) -> float:
        """Calculate similarity between songs based on content features."""
        # Combine all text features for TF-IDF
        features1 = [
            song1.get('title', ''),
            song1.get('artist', ''),
            song1.get('genre', ''),
            song1.get('album', ''),
            ' '.join(song1.get('tags', [])),
            song1.get('language', '')
        ]
        features2 = [
            song2.get('title', ''),
            song2.get('artist', ''),
            song2.get('genre', ''),
            song2.get('album', ''),
            ' '.join(song2.get('tags', [])),
            song2.get('language', '')
        ]

        # Calculate TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform([' '.join(features1), ' '.join(features2)])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _get_user_listening_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's listening history from reactions and views."""
        try:
            # Get user's reactions
            reactions = self.es.search(
                index="reaction",
                body={
                    "query": {
                        "term": {"reactorName.keyword": user_id}
                    }
                }
            )["hits"]["hits"]

            # Get songs the user has reacted to
            history = []
            for reaction in reactions:
                song_id = reaction["_source"]["songId"]
                song_data = self._get_song_data(song_id)
                if song_data:
                    history.append(song_data)

            return history
        except Exception as e:
            logger.error(f"Error getting user history: {str(e)}")
            return []

    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get content-based recommendations for a user."""
        try:
            # Get user data and history
            user_data = self._get_user_data(user_id)
            if not user_data:
                return []

            # Get user's favorite songs and listening history
            favorite_songs = []
            for song_id in user_data.get('favorites', []):
                song_data = self._get_song_data(song_id)
                if song_data:
                    favorite_songs.append(song_data)

            listening_history = self._get_user_listening_history(user_id)

            # Get all songs
            all_songs = self.es.search(
                index="songs",
                body={
                    "size": 1000,
                    "query": {"match_all": {}}
                }
            )["hits"]["hits"]

            # Calculate scores for each song
            song_scores = []
            for song in all_songs:
                song_data = song["_source"]
                song_id = song["_id"]

                # Skip if song is already in favorites
                if song_id in user_data.get('favorites', []):
                    continue

                # Calculate similarity with favorite songs
                favorite_similarity = max(
                    self._calculate_song_similarity(song_data, fav_song)
                    for fav_song in favorite_songs
                ) if favorite_songs else 0.0

                # Calculate similarity with listening history
                history_similarity = max(
                    self._calculate_song_similarity(song_data, hist_song)
                    for hist_song in listening_history
                ) if listening_history else 0.0

                # Combine scores (weighted average)
                final_score = 0.7 * favorite_similarity + 0.3 * history_similarity
                song_scores.append((song_data, final_score))

            # Sort by score and return top recommendations
            song_scores.sort(key=lambda x: x[1], reverse=True)
            return [song for song, _ in song_scores[:limit]]

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []