import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigSettings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    # Qdrant Settings
    QDRANT_HOST: str
    QDRANT_PORT: int  

    # Vector DB Settings
    VECTOR_COLLECTION_NAME: str
    VECTOR_DISTANCE_METRIC: str
    
    # Embedding Settings
    EMBEDDING_MODEL: str
    EMBEDDING_DIMENSION: int
    
    # API Keys
    GEMINI_API_KEY: str
    GROQ_API_KEY: str
    ELEVENLABS_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )

def get_settings():
    return ConfigSettings()