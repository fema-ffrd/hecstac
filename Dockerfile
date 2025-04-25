FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install git and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone rasqc (feature/stac-checker branch)
RUN git clone https://github.com/fema-ffrd/rasqc.git && \
    cd rasqc/ && \
    git checkout feature/stac-checker

# Install build tools
RUN pip install --no-cache-dir build

# Build and install rasqc
RUN cd /app/rasqc/ && \
    python -m build && \
    pip install --no-cache-dir dist/rasqc-0.0.1rc1-py3-none-any.whl

# Build and install hecstac
COPY . /app/hecstac/
RUN cd /app/hecstac/ && \
    python -m build && \
    pip install --no-cache-dir dist/hecstac-0.1.0rc3-py3-none-any.whl

# Default command (todo: change to entrypoint script)
CMD ["python"]