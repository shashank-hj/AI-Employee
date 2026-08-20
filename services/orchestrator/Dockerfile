FROM python:3.12-slim
WORKDIR /app
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/
RUN --mount=type=cache,target=/root/.cache/pip pip install uv
COPY shared/ /shared/
COPY services/orchestrator/ /app/
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system /shared/auth/ /shared/events/ /shared/guardrails/ /shared/llm/ /shared/models/ /shared/queue/ /shared/schemas/ /shared/usage/ /shared/utils/ .
EXPOSE 8001
CMD ["python", "-m", "orchestrator.main"]
