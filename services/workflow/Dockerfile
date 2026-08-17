FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY shared/ /shared/
COPY services/workflow/ /app/
RUN uv pip install --system /shared/auth/ /shared/events/ /shared/guardrails/ /shared/llm/ /shared/models/ /shared/queue/ /shared/schemas/ /shared/usage/ /shared/utils/ .
EXPOSE 8005
CMD ["python", "-m", "workflow.main"]
