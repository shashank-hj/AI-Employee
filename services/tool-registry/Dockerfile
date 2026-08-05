FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/tool-registry/ /app/
RUN uv pip install --system /shared/*/ .
EXPOSE 8002
CMD ["python", "-m", "tool_registry.main"]
