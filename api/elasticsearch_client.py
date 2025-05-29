from elasticsearch import Elasticsearch
from typing import List, Dict, Optional
from models.models import Song, Reaction, User
from conf.conf import settings
import logging

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    def __init__(self):
        try:
            self.es = Elasticsearch([{
                'host': settings.ELASTICSEARCH_HOST,
                'port': settings.ELASTICSEARCH_PORT
            }])
            # Test the connection
            if not self.es.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            logger.info("Successfully connected to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to initialize Elasticsearch client: {str(e)}")
            raise

    def get_all_songs(self) -> List[Song]:
        try:
            query = {
                "query": {"match_all": {}},
                "size": 1000
            }
            response = self.es.search(index="songs", body=query)
            return [Song(**hit["_source"]) for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error getting all songs: {str(e)}")
            raise

    def get_song_by_id(self, song_id: str) -> Song:
        try:
            response = self.es.get(index="songs", id=song_id)
            return Song(**response["_source"])
        except Exception as e:
            logger.error(f"Error getting song by ID {song_id}: {str(e)}")
            raise

    def get_song_reactions(self, song_id: str) -> List[Reaction]:
        try:
            query = {
                "query": {
                    "term": {"songId.keyword": song_id}
                }
            }
            response = self.es.search(index="reaction", body=query)
            return [Reaction(**hit["_source"]) for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error getting reactions for song {song_id}: {str(e)}")
            raise

    def get_user_reactions(self, username: str) -> List[Reaction]:
        try:
            query = {
                "query": {
                    "term": {"reactorName.keyword": username}
                }
            }
            response = self.es.search(index="reaction", body=query)
            return [Reaction(**hit["_source"]) for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error getting reactions for user {username}: {str(e)}")
            raise

    def get_user_by_id(self, user_id: str) -> User:
        try:
            response = self.es.get(index="users", id=user_id)
            return User(**response["_source"])
        except Exception as e:
            logger.error(f"Error getting user by ID {user_id}: {str(e)}")
            raise 