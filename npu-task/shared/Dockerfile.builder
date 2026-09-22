FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates ccache cmake git pkg-config python3 && \
    rm -rf /var/lib/apt/lists/*
