# Installation

This document explains how to install the application on Windows and macOS.

## Prerequisites

- Windows 10/11 or macOS (Intel / Apple Silicon).
- Docker Desktop installed and running.
- Google Drive installed and synchronized locally.
- A valid OpenAI API key.
- Internet access for the OpenAI API.

## Installation steps

1. Copy the project folder to the target machine.
2. Initialize the environment:
   - **Windows**: Run `scripts\setup.ps1` via PowerShell.
   - **macOS**: Run `scripts/setup.sh` via Terminal.
3. Open `.env` at the project root and fill in:
   - `OPENAI_API_KEY=...`
   - `RAW_DIR=...` (local absolute path to the synchronized Google Drive folder to index)
4. Create the quick-launch shortcut:
   - **Windows**: Run `scripts\create_shortcut.ps1` to add the app to the Start menu.
5. Launch the application:
   - **Windows**: Open the shortcut, or run `scripts\launch_app.vbs` or `make docker-up`.
   - **macOS**: Run `scripts/launch_app.sh` or `make docker-up`.

## Quick verification

- The application should open in your default web browser (`http://localhost:8501`).
- The "Conversation" and "Mise à jour" tabs should appear.

## Notes

- If the project folder is moved, recreate your shortcut.
- After a dependency update (pyproject.toml), run `make docker-install` to rebuild the Docker image.
