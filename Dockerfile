# 1. Set the base image
FROM python:3.12-slim

# 2. Set the environment variables
# - PYTHONUNBUFFERED=1 : Force Python to print all output to the console without buffering
# - DEBIAN_FRONTEND=noninteractive : Prevents interactive prompts during package installation
# - PYTHONPATH=/app : Sets Python root directory to /app
ENV PYTHONBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    SOFFICE_PATH=/usr/bin/soffice

# 3. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory
WORKDIR /app

# 5. Copy the pyproject.toml file to the working directory and install the package
COPY pyproject.toml ./
RUN python -m pip install --upgrade pip && \
    python -m pip install .

# 6. Copy the source code to the working directory
COPY src/ ./src

# 7. Expose the port that the application will run on
EXPOSE 8501

# 8. Set the command to run the application
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
