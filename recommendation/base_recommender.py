from abc import ABC, abstractmethod
from typing import List, Dict
from models.models import Song
from api.elasticsearch_client import ElasticsearchClient

class BaseRecommender(ABC):
    def __init__(self, es_client: ElasticsearchClient):
        self.es_client = es_client

    @abstractmethod
    def get_recommendations(self, user_id: str, limit: int = 10) -> List[Song]:
        pass