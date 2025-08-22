# build image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3 \
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
    && python -m build

# production image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# copy & install wheel from build stage
COPY --from=build /app/hecstac/dist/hecstac-*-py3-none-any.whl /app/
RUN pip install --no-cache-dir /app/hecstac-*-py3-none-any.whl \
    && rm -rf ./*.whl

COPY --from=build /app/hecstac/workflows /app