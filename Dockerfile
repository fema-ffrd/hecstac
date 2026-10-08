# build image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# gdal image requires venv
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH="/opt/venv/bin:$PATH"

# clone and build hecstac
COPY . /app/hecstac/
WORKDIR /app/hecstac/
RUN pip install --no-cache-dir build \
    && python -m build \
    && pip wheel --no-cache-dir --wheel-dir /wheelhouse dist/hecstac-*-py3-none-any.whl

# production image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

WORKDIR /app

COPY --from=build /wheelhouse /wheelhouse

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-index --find-links=/wheelhouse hecstac \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /wheelhouse

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=build /app/hecstac/workflows /app