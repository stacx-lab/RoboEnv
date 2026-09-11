FROM ubuntu:24.04
# Bootstrap Ubuntu's trust roots so package downloads can use verified HTTPS.
ADD --checksum=sha256:641de77d8f142cfd62a1a6f964ba67b20754d3337c480efb529d086075a06c9a \
    https://archive.ubuntu.com/ubuntu/pool/main/c/ca-certificates/ca-certificates_20240203_all.deb /tmp/ca.deb
RUN mkdir -p /etc/ssl/certs \
    && dpkg-deb -x /tmp/ca.deb /tmp/ca \
    && cat /tmp/ca/usr/share/ca-certificates/mozilla/*.crt > /etc/ssl/certs/ca-certificates.crt \
    && sed -i 's|http://|https://|g' /etc/apt/sources.list.d/ubuntu.sources \
    && printf 'Acquire::Retries "3";\nAcquire::https::Timeout "15";\n' \
      > /etc/apt/apt.conf.d/80-retries \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-numpy python3-yaml ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* /tmp/ca /tmp/ca.deb
COPY sdk/ /tmp/sdk/
RUN python3 -m venv --system-site-packages /opt/roboenv-sdk \
    && /opt/roboenv-sdk/bin/pip install --no-deps /tmp/sdk/*.whl
ENV PATH=/opt/roboenv-sdk/bin:$PATH \
    ROBOENV_ROBOT=/env/robot ROBOENV_MANIFEST=/env/environment.json \
    ROBOENV_SOCKET=/run/roboenv/sim.sock ROBOENV_SESSION=harbor ROBOENV_RUN_DIR=/run/roboenv/run
COPY workspace/ /app/
WORKDIR /app
CMD ["sleep", "infinity"]
