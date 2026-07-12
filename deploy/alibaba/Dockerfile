FROM python:3.11-slim

# Install Node.js 18 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg nginx supervisor \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source
COPY . .

# Build CLI
RUN cd cli && npm install && npm run build

# Build frontend
RUN cd frontend && npm install && npx vite build

# Nginx config for frontend
RUN echo 'server { \
    listen 3000; \
    root /app/frontend/dist; \
    index index.html; \
    location / { try_files $uri $uri/ /index.html; } \
    location /api/ { proxy_pass http://127.0.0.1:8000/api/; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; proxy_buffering off; } \
}' > /etc/nginx/sites-available/synthclaw \
    && ln -sf /etc/nginx/sites-available/synthclaw /etc/nginx/sites-enabled/default

# Supervisor to run both Python API + nginx
RUN echo '[supervisord]\nnodaemon=true\n\n[program:agent]\ncommand=python3 /app/main.py\ndirectory=/app\nautorestart=true\nstdout_logfile=/dev/stdout\nstdout_logfile_maxbytes=0\nstderr_logfile=/dev/stderr\nstderr_logfile_maxbytes=0\n\n[program:nginx]\ncommand=nginx -g "daemon off;"\nautorestart=true' > /etc/supervisor/conf.d/synthclaw.conf

EXPOSE 3000 8000

ENV SYNTHCLAW_BASE_DIR=/app
ENV SYNTHCLAW_API_HOST=0.0.0.0
ENV SYNTHCLAW_API_PORT=8000

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]
