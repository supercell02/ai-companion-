from pydantic_settings import BaseSettings

class Config(BaseSettings):
    openai_api_key: str
    db_path: str = "companion_memory.db"
    model_response: str = "gpt-4o-mini"
    model_logic: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"

config = Config()