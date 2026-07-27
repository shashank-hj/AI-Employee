#!/bin/bash
set -e
echo "Stopping AI Employee Platform..."
docker compose --profile all down
echo "All services stopped."
