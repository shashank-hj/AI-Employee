FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv pip install --system .
COPY . .
EXPOSE 8001
CMD ["python", "-m", "orchestrator.main"]
