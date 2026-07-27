import uvicorn
from tool_registry.config import settings
from tool_registry.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("tool_registry.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG, log_level=settings.LOG_LEVEL.lower())
