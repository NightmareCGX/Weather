#!/bin/sh
# Generate a self-signed TLS certificate for the weather gateway when none is
# mounted. nginx:alpine includes openssl and runs /docker-entrypoint.d/*.sh
# before starting the server. Production deployments should mount a real
# certificate (or terminate TLS upstream) instead of relying on this.
set -e

CERT_DIR="/etc/nginx/certs"
CERT_FILE="${CERT_DIR}/tls.crt"
KEY_FILE="${CERT_DIR}/tls.key"

if [ -f "${CERT_FILE}" ] && [ -f "${KEY_FILE}" ]; then
    echo "gateway: using existing TLS certificate from ${CERT_DIR}"
    exit 0
fi

mkdir -p "${CERT_DIR}"
echo "gateway: generating self-signed TLS certificate (CN=weather-gateway)"
openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
    -keyout "${KEY_FILE}" -out "${CERT_FILE}" \
    -subj "/CN=weather-gateway" \
    -addext "subjectAltName=DNS:localhost,DNS:weather-gateway,IP:127.0.0.1"
chmod 600 "${KEY_FILE}"
