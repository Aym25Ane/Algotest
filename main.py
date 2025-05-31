from fastapi import FastAPI, HTTPException
from models.models import RecommendationRequest, Song
from api.elasticsearch_client import ElasticsearchClient
from recommendation.content_based import ContentBasedRecommender
from recommendation.collaborative import CollaborativeRecommender
from recommendation.popularity import PopularityRecommender
from recommendation.hybrid_recommender import HybridRecommender
from conf.conf import settings
import logging
from typing import List
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Recommendation System")

# Initialize Elasticsearch client
try:
    es_client = ElasticsearchClient()
    logger.info("Successfully connected to Elasticsearch")
except Exception as e:
    logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
    raise

# Initialize recommenders
try:
    content_recommender = ContentBasedRecommender(es_client)
    collab_recommender = CollaborativeRecommender(es_client)
    popularity_recommender = PopularityRecommender(es_client)
    hybrid_recommender = HybridRecommender(es_client)
    logger.info("Successfully initialized all recommenders")
except Exception as e:
    logger.error(f"Failed to initialize recommenders: {str(e)}")
    raise

@app.post("/api/recommendations/hybrid")
async def hybrid_recommendations(request: RecommendationRequest) -> List[Song]:
    try:
        logger.info(f"Processing hybrid recommendations for user: {request.userId}")
        recommendations = hybrid_recommender.get_recommendations(request.userId, request.limit)
        logger.info(f"Successfully generated {len(recommendations)} hybrid recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/content-based")
async def content_based_recommendations(request: RecommendationRequest) -> List[Song]:
    try:
        logger.info(f"Processing content-based recommendations for user: {request.userId}")
        recommendations = content_recommender.get_recommendations(request.userId, request.limit)
        logger.info(f"Successfully generated {len(recommendations)} content-based recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Error in content-based recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/collaborative")
async def collaborative_recommendations(request: RecommendationRequest) -> List[Song]:
    try:
        logger.info(f"Processing collaborative recommendations for user: {request.userId}")
        recommendations = collab_recommender.get_recommendations(request.userId, request.limit)
        logger.info(f"Successfully generated {len(recommendations)} collaborative recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Error in collaborative recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations/popularity")
async def popularity_recommendations(request: RecommendationRequest) -> List[Song]:
    try:
        logger.info(f"Processing popularity recommendations for user: {request.userId}")
        recommendations = popularity_recommender.get_recommendations(request.userId, request.limit)
        logger.info(f"Successfully generated {len(recommendations)} popularity recommendations")
        return recommendations
    except Exception as e:
        logger.error(f"Error in popularity recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.RECOMMENDATION_SERVICE_PORT) 