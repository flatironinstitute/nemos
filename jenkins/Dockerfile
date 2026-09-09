FROM docker.io/ubuntu:24.04

# Kept deliberately independent of the nemos source tree: this image only needs
# rebuilding when these system dependencies change, not on every commit.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3.12 \
      python3.12-venv \
      ruby \
      ruby-dev \
      build-essential \
      libcurl4 \
      git \
      curl \
      jq \
      && \
    apt-get autoremove --purge -y && \
    apt-get autoclean -y && \
    rm -rf /var/cache/apt/* /var/lib/apt/lists/*

# Same checker and floor version the Read the Docs build used, so link-check
# behaviour does not change when the build moves here. Its async dependency
# chain builds a native extension (io-event), which is why ruby-dev and
# build-essential are installed above.
RUN gem install html-proofer -v ">= 5.0.9"
