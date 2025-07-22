# gunicorn_config.py
bind = "0.0.0.0:8062"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"