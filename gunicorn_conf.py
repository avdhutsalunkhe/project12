import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

workers = 1  # Strictly 1 worker for Render free tier (512MB RAM)
preload_app = False  # Ensure lazy loading inside worker

worker_class = "uvicorn.workers.UvicornWorker"

timeout = 120
keepalive = 5

accesslog = "-"
errorlog = "-"
