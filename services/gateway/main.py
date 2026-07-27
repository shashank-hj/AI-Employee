import uvicorn
from gateway.config import settings
from gateway.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("gateway.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG, log_level=settings.LOG_LEVEL.lower())
