from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
import logging
from sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class PopularityRecommender:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
        self.sentiment_analyzer = SentimentAnalyzer()

    def _get_song_data(self, song_id: str) -> Dict[str, Any]:
        """Get song data including metadata."""
        try:
            song = self.es.get(index="songs", id=song_id)
            return song["_source"]
        except Exception as e:
            logger.error(f"Error fetching song data: {str(e)}")
            return {}

    def _calculate_popularity_score(self, song: Dict[str, Any]) -> float:
        """Calculate popularity score based on multiple metrics including sentiment analysis."""
        try:
            # Get engagement metrics
            view_count = song.get('viewCount', 0)
            reaction_count = song.get('totalReactionCount', 0)
            comment_count = song.get('commentCount', 0)

            # Calculate time decay factor
            release_date = song.get('releaseDate')
            if release_date:
                release_date = datetime.fromisoformat(release_date.replace('Z', '+00:00'))
                days_old = (datetime.now() - release_date).days
                time_decay = 1.0 / (1.0 + np.log1p(days_old))  # Logarithmic decay
            else:
                time_decay = 0.5  # Default for songs without release date

            # Calculate sentiment score
            comments = song.get('comments', [])
            sentiment_score = 0.5  # Default neutral score
            if comments:
                sentiment_stats = self.sentiment_analyzer.analyze_comments(comments)
                # Normalize sentiment score (1-5 scale to 0-1)
                sentiment_score = (sentiment_stats['average_sentiment'] - 1) / 4

            # Calculate weighted score
            weights = {
                'views': 0.3,
                'reactions': 0.2,
                'comments': 0.2,
                'sentiment': 0.2,
                'time_decay': 0.1
            }

            # Normalize metrics
            max_views = 10000  # Example max value
            max_reactions = 1000
            max_comments = 500

            normalized_views = min(view_count / max_views, 1.0)
            normalized_reactions = min(reaction_count / max_reactions, 1.0)
            normalized_comments = min(comment_count / max_comments, 1.0)

            # Calculate final score
            score = (
                    weights['views'] * normalized_views +
                    weights['reactions'] * normalized_reactions +
                    weights['comments'] * normalized_comments +
                    weights['sentiment'] * sentiment_score +
                    weights['time_decay'] * time_decay
            )

            return float(score)

        except Exception as e:
            logger.error(f"Error calculating popularity score: {str(e)}")
            return 0.0

    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get popularity-based recommendations."""
        try:
            # Get all songs
            all_songs = self.es.search(
                index="songs",
                body={
                    "size": 1000,
                    "query": {"match_all": {}}
                }
            )["hits"]["hits"]

            # Calculate popularity scores
            song_scores = []
            for song in all_songs:
                song_data = song["_source"]
                score = self._calculate_popularity_score(song_data)
                song_scores.append((song_data, score))

            # Sort by score and return top recommendations
            song_scores.sort(key=lambda x: x[1], reverse=True)
            return [song for song, _ in song_scores[:limit]]

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return [],