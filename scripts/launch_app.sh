#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/.." || exit

if ! docker info > /dev/null 2>&1; then
    open -a "Docker"
    while ! docker info > /dev/null 2>&1; do sleep 5; done
fi

docker compose up -d
sleep 2
open http://localhost:8501
