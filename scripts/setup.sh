#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "\033[0;36m--- Configuration de l'Assistant Documentaire RAG ---\033[0m"

if [ ! -f .env ]; then
    echo -e "\033[0;33m[INFO] Création du fichier .env à partir de .env.example...\033[0m"

    cp .env.example .env

    echo -e "\033[0;32m[OK] Fichier .env créé. Veuillez l'ouvrir pour remplir vos accès !\033[0m"
else
    echo "[INFO] Fichier .env déjà existant. Étape ignorée."
fi

echo -e "\033[0;33m\n[INFO] Initialisation de l'environnement Docker...\033[0m"
if ! docker info > /dev/null 2>&1; then
    echo -e "\033[0;31m[ATTENTION] Docker Desktop n'est pas lancé. Allumez-le pour compiler l'application.\033[0m"
else
    docker compose build
    echo -e "\033[0;32m[OK] Image Docker compilée avec succès !\033[0m"
fi

echo -e "\033[0;32m\n[SUCCÈS] Installation terminée !\033[0m"
