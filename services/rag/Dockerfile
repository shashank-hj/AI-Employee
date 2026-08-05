FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/rag/ /app/
RUN uv pip install --system /shared/*/ .
EXPOSE 8004
CMD ["python", "-m", "rag.main"]
