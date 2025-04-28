# build image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest AS build

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    python3-venv \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# gdal image requires venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# clone and build rasqc && hecstac
RUN git clone https://github.com/fema-ffrd/rasqc.git && \
    cd rasqc/ && \
    git checkout feature/stac-checker && \
    pip install --no-cache-dir build && \
    python -m build

COPY . /app/hecstac/
RUN cd /app/hecstac/ && \
    pip install --no-cache-dir build && \
    python -m build

# production image
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy wheels from build stage
COPY --from=build /app/rasqc/dist/rasqc-0.0.1rc1-py3-none-any.whl /app/
COPY --from=build /app/hecstac/dist/hecstac-*-py3-none-any.whl /app/

# install wheels
RUN pip install --no-cache-dir \
    /app/rasqc-0.0.1rc1-py3-none-any.whl \
    /app/hecstac-*-py3-none-any.whl

RUN rm -rf *.whl
COPY --from=build /app/hecstac/workflows /app