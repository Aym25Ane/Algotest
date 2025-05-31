from pydantic import BaseSettings


class Settings(BaseSettings):
    # Elasticsearch settings
    ELASTICSEARCH_HOST: str = "localhost"
    ELASTICSEARCH_PORT: int = 9200

    # Recommendation service settings
    RECOMMENDATION_SERVICE_PORT: int = 8000

    # Spring Boot application settings
    SPRING_APP_NAME: str = "PFA_Back"
    SPRING_SERVER_PORT: int = 8080
    FILE_BASE_URL: str = "http://localhost:8080"
    UPLOAD_DIR_AUDIO: str = "uploads/audio"
    UPLOAD_DIR_IMAGES: str = "uploads/images"

    class Config:
        env_file = ".env"


settings = Settings()