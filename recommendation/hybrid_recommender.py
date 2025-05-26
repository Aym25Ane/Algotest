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
        self.collaborative_recommender = CollaborativeRecommender(es_client)
        self.popularity_recommender = PopularityRecommender(es_client)

        # Poids par défaut
        self.default_weights = {
            'popularity': 0.3,
            'content': 0.3,
            'collaborative': 0.4
        }

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
                return self.default_weights

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
                return self.default_weights

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
            return self.default_weights

    def get_recommendations(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining all algorithms"""
        try:
            # Get recommendations from each algorithm
            popularity_recs = self.popularity_recommender.get_recommendations(user_id)
            content_recs = self.content_recommender.get_recommendations(user_id)
            collaborative_recs = self.collaborative_recommender.get_recommendations(user_id)

            # Calculate user-specific weights
            weights = self._calculate_user_weights(user_id)

            # Combine recommendations with weights
            combined_scores = {}

            # Process popularity recommendations
            for rec in popularity_recs:
                song_id = rec['id']
                combined_scores[song_id] = {
                    'score': rec['score'] * weights['popularity'],
                    'metadata': rec
                }

            # Process content-based recommendations
            for rec in content_recs:
                song_id = rec['id']
                if song_id in combined_scores:
                    combined_scores[song_id]['score'] += rec['score'] * weights['content']
                else:
                    combined_scores[song_id] = {
                        'score': rec['score'] * weights['content'],
                        'metadata': rec
                    }

            # Process collaborative recommendations
            for rec in collaborative_recs:
                song_id = rec['id']
                if song_id in combined_scores:
                    combined_scores[song_id]['score'] += rec['score'] * weights['collaborative']
                else:
                    combined_scores[song_id] = {
                        'score': rec['score'] * weights['collaborative'],
                        'metadata': rec
                    }

            # Sort by combined score
            sorted_recommendations = sorted(
                combined_scores.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            )[:limit]

            # Format final recommendations
            final_recommendations = []
            for song_id, data in sorted_recommendations:
                recommendation = data['metadata'].copy()
                recommendation['hybrid_score'] = data['score']
                final_recommendations.append(recommendation)

            return final_recommendations

        except Exception as e:
            logger.error(f"Error getting hybrid recommendations: {str(e)}")
            # Fallback to popularity recommendations
            return self.popularity_recommender.get_recommendations(user_id, limit)

    def get_recommendation_explanation(self, user_id: str, song_id: str) -> Dict[str, Any]:
        """Get explanation for why a song was recommended"""
        try:
            weights = self._calculate_user_weights(user_id)

            explanation = {
                'popularity_score': self.popularity_recommender.get_song_score(song_id),
                'content_score': self.content_recommender.get_song_score(user_id, song_id),
                'collaborative_score': self.collaborative_recommender.get_song_score(user_id, song_id),
                'weights': weights,
                'hybrid_score': 0.0
            }

            # Calculate hybrid score
            explanation['hybrid_score'] = (
                    explanation['popularity_score'] * weights['popularity'] +
                    explanation['content_score'] * weights['content'] +
                    explanation['collaborative_score'] * weights['collaborative']
            )

            return explanation

        except Exception as e:
            logger.error(f"Error getting recommendation explanation: {str(e)}")
            return {}