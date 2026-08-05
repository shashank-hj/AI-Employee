FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/speech/ /app/
RUN uv pip install --system /shared/*/ .
EXPOSE 8006
CMD ["python", "-m", "speech.main"]
