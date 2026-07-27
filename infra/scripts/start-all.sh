#!/bin/bash
set -e

echo "Starting AI Employee Platform..."
docker compose --profile all up -d
echo "All services started."
echo "Gateway: http://localhost:8000"
echo "Orchestrator: http://localhost:8001"
echo "Tool Registry: http://localhost:8002"
echo "Memory: http://localhost:8003"
echo "RAG: http://localhost:8004"
echo "Workflow: http://localhost:8005"
