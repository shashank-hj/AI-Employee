import uvicorn

from speech.config import settings
from speech.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "speech.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
