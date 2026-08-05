FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/orchestrator/ /app/
RUN uv pip install --system /shared/*/ .
EXPOSE 8001
CMD ["python", "-m", "orchestrator.main"]
