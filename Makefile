.PHONY: help

help:
	@echo "Available targets:"
	@echo "  make user-install   - Install in a virtual environment"
	@echo "  make install        - Install production dependencies"
	@echo "  make run            - Run the Streamlit app"
	@echo "  make test           - Run tests with pytest"
	@echo "  make docs           - Preview docs locally"
	@echo "  make build-docs     - Build static docs site"
	@echo "  make clean          - Remove build artifacts and cache (safe)"
	@echo "  make update-index   - Rebuild the document index"
	@echo "  make reset-index    - Destructive: delete all local data (requires confirmation)"
	@echo "  make docker-install - Install in Docker container"
	@echo "  make docker-up      - Run the Streamlit app in Docker container"
	@echo "  make docker-down    - Shutdown the Docker container"
	@echo "  make docker-restart - Relaunch the app in Docker container"
	@echo "  make docker-clean   - Clean orphaned Docker images to free up disk space"

# --- Local Development Environment ---

.PHONY: user-install install run test docs build-docs clean reset-index update-index

# Automatically use .venv if it exists, otherwise defaults to system Python

VENV = .venv
VENV_ACTIVATE = $(shell test -d $(VENV) && find $(VENV) -name "activate")
VENV_PYTHON = $(shell test -d $(VENV) && . $(VENV_ACTIVATE) 2>/dev/null; command -v python 2>/dev/null || where python 2>/dev/null)
SYSTEM_PYTHON = $(shell command -v python 2>/dev/null || where python 2>/dev/null)
PYTHON = $(or $(VENV_PYTHON), $(SYSTEM_PYTHON), "PYTHONNOTFOUND")

# Install env and requirements

PACKAGE = $(shell grep "^name" pyproject.toml | awk -F'"' '{print $2}')
BUILD_CACHE = $(PACKAGE).egg-info

user-install: | $(VENV) $(BUILD_CACHE)

install: $(BUILD_CACHE)

$(VENV):
	$(SYSTEM_PYTHON) -m venv $(VENV)

$(BUILD_CACHE): pyproject.toml
	@echo "Installing dependencies on $(PYTHON)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .

# Mkdocs commands for documentation

docs:
	mkdocs serve

build-docs:
	mkdocs build

# Run the Streamlit app

run:
	streamlit run src/ui/app.py

# Run tests with pytest

test:
	$(PYTHON) -m pytest

# Commands for index management

update-index:
	$(PYTHON) -m src.core.update_pipeline

reset-index:
	@echo "⚠️  WARNING: This will delete all local data (registry, chunks, index, logs)."
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		$(PYTHON) -m src.admin.reset_local_state --yes; \
		echo "✓ Local state reset."; \
	else \
		echo "Cancelled."; \
	fi

# Clean build artifacts, test artifacts, and caches

clean:
	rm -rf build/ dist/ src/*.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/ site/
	rm -rf .coverage .coverage.* htmlcov/ coverage.xml .hypothesis/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# --- Docker Development Environment ---

.PHONY: docker-install docker-up docker-down docker-restart docker-clean

# First-time installation or major update (dependencies)
docker-install:
	docker compose up -d --build

# Quick startup (without rebuilding)
docker-up:
	docker compose up -d

# Shutdown of containers
docker-down:
	docker compose down

# Quick relaunch
docker-restart:
	docker compose down && docker compose up -d

# Cleanup of orphaned Docker images to free up disk space
docker-clean:
	docker compose down
	docker image prune -f
