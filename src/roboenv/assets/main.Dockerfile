FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-numpy python3-yaml ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY sdk/ /tmp/sdk/
RUN python3 -m venv --system-site-packages /opt/roboenv-sdk \
    && /opt/roboenv-sdk/bin/pip install --no-deps /tmp/sdk/*.whl
ENV PATH=/opt/roboenv-sdk/bin:$PATH \
    ROBOENV_ROBOT=/env/robot ROBOENV_MANIFEST=/env/environment.json \
    ROBOENV_SOCKET=/run/roboenv/sim.sock ROBOENV_SESSION=harbor ROBOENV_RUN_DIR=/run/roboenv/run
COPY workspace/ /app/
WORKDIR /app
CMD ["sleep", "infinity"]
