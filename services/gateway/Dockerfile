FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/gateway/ /app/
RUN uv pip install --system /shared/*/ .
EXPOSE 8000
CMD ["python", "-m", "gateway.main"]
