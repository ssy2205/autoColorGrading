import os

# Render aiohttp server port
port = int(os.environ.get("PORT", 5001))

# Gunicorn config variables
bind = f"0.0.0.0:{port}"
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
