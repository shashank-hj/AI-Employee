import uvicorn
from orchestrator.config import settings
from orchestrator.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("orchestrator.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG, log_level=settings.LOG_LEVEL.lower())
