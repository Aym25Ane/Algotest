from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
import logging
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

logger = logging.getLogger(__name__)


class CollaborativeRecommender:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client

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

    def _get_user_reactions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all reactions from a user."""
        try:
            reactions = self.es.search(
                index="reaction",
                body={
                    "query": {
                        "term": {"reactorName.keyword": user_id}
                    }
                }
            )["hits"]["hits"]
            return [r["_source"] for r in reactions]
        except Exception as e:
            logger.error(f"Error getting user reactions: {str(e)}")
            return []

    def _get_similar_users(self, user_id: str, user_reactions: List[Dict[str, Any]]) -> List[str]:
        """Find users with similar taste based on reactions."""
        try:
            # Get all reactions
            all_reactions = self.es.search(
                index="reaction",
                body={
                    "size": 1000,
                    "query": {"match_all": {}}
                }
            )["hits"]["hits"]

            # Create user-song reaction matrix
            user_song_matrix = defaultdict(dict)
            for reaction in all_reactions:
                r = reaction["_source"]
                user_song_matrix[r["reactorName"]][r["songId"]] = 1

            # Convert to numpy array for similarity calculation
            users = list(user_song_matrix.keys())
            songs = list(set(song for reactions in user_song_matrix.values() for song in reactions))

            # Create user vectors
            user_vectors = []
            for user in users:
                vector = [user_song_matrix[user].get(song, 0) for song in songs]
                user_vectors.append(vector)

            # Calculate user similarity
            user_vectors = np.array(user_vectors)
            similarity_matrix = cosine_similarity(user_vectors)

            # Find similar users
            user_idx = users.index(user_id)
            similar_users = []
            for idx, similarity in enumerate(similarity_matrix[user_idx]):
                if idx != user_idx and similarity > 0.3:  # Threshold for similarity
                    similar_users.append((users[idx], similarity))

            # Sort by similarity and return top users
            similar_users.sort(key=lambda x: x[1], reverse=True)
            return [user for user, _ in similar_users[:10]]

        except Exception as e:
            logger.error(f"Error finding similar users: {str(e)}")
            return []

    def _get_recommendations_from_similar_users(self, similar_users: List[str], user_id: str) -> List[Dict[str, Any]]:
        """Get song recommendations from similar users."""
        try:
            # Get songs liked by similar users
            recommended_songs = defaultdict(float)
            user_data = self._get_user_data(user_id)
            user_favorites = set(user_data.get('favorites', []))

            for similar_user in similar_users:
                # Get reactions from similar user
                reactions = self._get_user_reactions(similar_user)

                # Add songs to recommendations
                for reaction in reactions:
                    song_id = reaction["songId"]
                    if song_id not in user_favorites:  # Don't recommend songs user already likes
                        recommended_songs[song_id] += 1.0

            # Convert to list and sort by score
            song_scores = [(song_id, score) for song_id, score in recommended_songs.items()]
            song_scores.sort(key=lambda x: x[1], reverse=True)

            # Get song details
            recommendations = []
            for song_id, _ in song_scores:
                song_data = self._get_song_data(song_id)
                if song_data:
                    recommendations.append(song_data)

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations from similar users: {str(e)}")
            return []

    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get collaborative recommendations for a user."""
        try:
            # Get user's reactions
            user_reactions = self._get_user_reactions(user_id)
            if not user_reactions:
                return []

            # Find similar users
            similar_users = self._get_similar_users(user_id, user_reactions)
            if not similar_users:
                return []

            # Get recommendations from similar users
            recommendations = self._get_recommendations_from_similar_users(similar_users, user_id)
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [] 