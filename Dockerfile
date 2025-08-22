FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    postgresql-client \
    libpq-dev \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    proj-bin \
    proj-data \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    git \
    curl \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify GDAL installation and set correct paths
RUN ldconfig \
    && find /usr -name "*gdal*" -type f 2>/dev/null | head -10 \
    && find /usr -name "*geos*" -type f 2>/dev/null | head -10

# Set GDAL library paths
ENV GDAL_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgdal.so
ENV GEOS_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libgeos_c.so
ENV PROJ_LIB=/usr/share/proj
ENV GDAL_DATA=/usr/share/gdal

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Set working directory to config (where manage.py is located)
WORKDIR /app/config

# Create directories for static files and media
RUN mkdir -p /app/staticfiles /app/media

# Collect static files (will be overridden in docker-compose)
RUN python manage.py collectstatic --noinput --clear || true

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check (adjust path if needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/admin/login/ || exit 1

# Default command (run from config directory)
CMD ["uvicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout", "120", "config.asgi:application"]