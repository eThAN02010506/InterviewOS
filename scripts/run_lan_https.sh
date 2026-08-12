#!/usr/bin/env bash
set -euo pipefail

LAN_ADDRESS="${1:-}"
HTTPS_PORT="${INTERVIEW_OS_HTTPS_PORT:-8443}"
TLS_DIR="${INTERVIEW_OS_TLS_DIR:-data/tls}"

if [[ -z "${LAN_ADDRESS}" ]]; then
  echo "Usage: scripts/run_lan_https.sh <LAN IPv4 address>" >&2
  echo "Example: scripts/run_lan_https.sh 192.168.1.16" >&2
  exit 2
fi
if [[ ! "${LAN_ADDRESS}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "LAN address must be an IPv4 address, got: ${LAN_ADDRESS}" >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; create the project virtual environment first." >&2
  exit 2
fi

mkdir -p "${TLS_DIR}"
CA_KEY="${TLS_DIR}/interview-os-local-ca.key"
CA_CERT="${TLS_DIR}/interview-os-local-ca.crt"
SERVER_KEY="${TLS_DIR}/interview-os-lan.key"
SERVER_CSR="${TLS_DIR}/interview-os-lan.csr"
SERVER_CERT="${TLS_DIR}/interview-os-lan.crt"
SERVER_EXT="${TLS_DIR}/interview-os-lan.ext"

if [[ ! -f "${CA_KEY}" || ! -f "${CA_CERT}" ]]; then
  openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
    -keyout "${CA_KEY}" -out "${CA_CERT}" \
    -subj "/CN=InterviewOS Local CA" \
    -addext "basicConstraints=critical,CA:TRUE" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"
fi

cat > "${SERVER_EXT}" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LAN_ADDRESS}
EOF

# Regenerate the leaf certificate on every launch so DHCP address changes do
# not leave users with a certificate for yesterday's address. The local CA is
# stable and is the only certificate client devices need to trust once.
openssl req -new -newkey rsa:2048 -nodes -keyout "${SERVER_KEY}" \
  -out "${SERVER_CSR}" -subj "/CN=${LAN_ADDRESS}"
openssl x509 -req -sha256 -days 365 -in "${SERVER_CSR}" \
  -CA "${CA_CERT}" -CAkey "${CA_KEY}" -CAcreateserial \
  -out "${SERVER_CERT}" -extfile "${SERVER_EXT}"
chmod 600 "${CA_KEY}" "${SERVER_KEY}"

echo "InterviewOS HTTPS: https://${LAN_ADDRESS}:${HTTPS_PORT}"
echo "Trust this CA certificate once on each client device: ${CA_CERT}"
exec .venv/bin/python -m uvicorn interview_os.api.app:app \
  --host 0.0.0.0 --port "${HTTPS_PORT}" \
  --ssl-keyfile "${SERVER_KEY}" --ssl-certfile "${SERVER_CERT}"
