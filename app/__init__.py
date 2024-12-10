# app/__init__.py
import os
from .main import app

# Load environment variables
os.environ.setdefault("APP_ENV", "development")

# Log environment
print(f"Running in {os.environ['APP_ENV']} mode.")
