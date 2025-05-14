from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
import logging
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender
from .popularity import PopularityRecommender

logger = logging.getLogger(__name__)


class HybridRecommender:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.content_recommender = ContentBasedRecommender(es_client)
        self.collab_recommender = CollaborativeRecommender(es_client)
        self.popularity_recommender = PopularityRecommender(es_client)

    def _get_user_data(self, user_id: str) -> Dict[str, Any]:
        """Get user data including profile and preferences."""
        try:
            user = self.es.get(index="users", id=user_id)
            return user["_source"]
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            return {}

    def _calculate_user_weights(self, user_id: str) -> Dict[str, float]:
        """Calculate weights for different recommendation approaches based on user behavior."""
        try:
            user_data = self._get_user_data(user_id)
            if not user_data:
                return {
                    'content': 0.4,
                    'collaborative': 0.3,
                    'popularity': 0.3
                }

            # Get user's activity metrics
            favorites_count = len(user_data.get('favorites', []))
            playlists_count = len(user_data.get('playlists', []))

            # Get user's reactions
            reactions = self.es.search(
                index="reaction",
                body={
                    "query": {
                        "term": {"reactorName.keyword": user_id}
                    }
                }
            )["hits"]["hits"]
            reactions_count = len(reactions)

            # Calculate weights based on user behavior
            total_activity = favorites_count + playlists_count + reactions_count
            if total_activity == 0:
                return {
                    'content': 0.4,
                    'collaborative': 0.3,
                    'popularity': 0.3
                }

            # More favorites/playlists -> higher content weight
            content_weight = 0.4 + (favorites_count + playlists_count) / (2 * total_activity)

            # More reactions -> higher collaborative weight
            collab_weight = 0.3 + reactions_count / total_activity

            # Adjust popularity weight
            popularity_weight = 1.0 - content_weight - collab_weight

            return {
                'content': content_weight,
                'collaborative': collab_weight,
                'popularity': popularity_weight
            }

        except Exception as e:
            logger.error(f"Error calculating user weights: {str(e)}")
            return {
                'content': 0.4,
                'collaborative': 0.3,
                'popularity': 0.3
            }

    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining all approaches."""
        try:
            # Get recommendations from each approach
            content_recs = self.content_recommender.get_recommendations(user_id, limit * 2)
            collab_recs = self.collab_recommender.get_recommendations(user_id, limit * 2)
            popularity_recs = self.popularity_recommender.get_recommendations(user_id, limit * 2)

            # Calculate weights for this user
            weights = self._calculate_user_weights(user_id)

            # Combine recommendations with weights
            song_scores = {}
            for song in content_recs:
                song_id = song.get('id')
                if song_id:
                    song_scores[song_id] = {
                        'song': song,
                        'score': weights['content']
                    }

            for song in collab_recs:
                song_id = song.get('id')
                if song_id:
                    if song_id in song_scores:
                        song_scores[song_id]['score'] += weights['collaborative']
                    else:
                        song_scores[song_id] = {
                            'song': song,
                            'score': weights['collaborative']
                        }

            for song in popularity_recs:
                song_id = song.get('id')
                if song_id:
                    if song_id in song_scores:
                        song_scores[song_id]['score'] += weights['popularity']
                    else:
                        song_scores[song_id] = {
                            'song': song,
                            'score': weights['popularity']
                        }

            # Sort by combined score
            sorted_songs = sorted(
                song_scores.values(),
                key=lambda x: x['score'],
                reverse=True
            )

            # Return top recommendations
            return [item['song'] for item in sorted_songs[:limit]]

        except Exception as e:
            logger.error(f"Error generating hybrid recommendations: {str(e)}")
            # Fallback to popularity recommendations
            return self.popularity_recommender.get_recommendations(user_id, limit) 