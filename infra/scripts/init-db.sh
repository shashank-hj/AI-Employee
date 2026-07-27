#!/bin/bash
set -e

echo "Waiting for PostgreSQL..."
until pg_isready -h "${DB_HOST:-postgres}" -p "${DB_PORT:-5432}" -U "${DB_USER:-postgres}"; do
    sleep 1
done
echo "PostgreSQL is ready."

echo "Running migrations..."
for service in gateway orchestrator tool-registry memory rag workflow; do
    if [ -d "services/$service" ]; then
        echo "Running migrations for $service..."
        cd "services/$service" && alembic upgrade head && cd ../..
    fi
done

echo "Database initialization complete."
