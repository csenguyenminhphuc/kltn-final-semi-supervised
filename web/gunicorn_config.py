# Gunicorn configuration file

# Bind address
bind = "0.0.0.0:12347"

# Number of worker processes
workers = 2

# Worker class
worker_class = "sync"

# Timeout for requests (seconds)
timeout = 120

# Keep alive
keepalive = 5

# Max requests per worker before restart
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "app3"

# Daemon mode (set to True to run in background)
daemon = False

# Reload on code changes (development only)
reload = False
