#!/usr/bin/env bash
# Fetch TLS certificates from nasbox for the SAM server.
# Run this after each Let's Encrypt renewal on nasbox (currently ~every 90 days).
# Cert: *.nasbox.nakomis.com  —  expires 2026-07-13
#
# Usage:
#   bash scripts/fetch_certs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$(dirname "$SCRIPT_DIR")/certs"
NASBOX="nasbox.local"
REMOTE_CERT_DIR="/etc/letsencrypt/live/nasbox.nakomis.com"

mkdir -p "$CERT_DIR"

echo "Fetching certificates from $NASBOX..."

# privkey.pem is root-owned on the nasbox; use sudo -n (non-interactive)
# You'll need NOPASSWD sudo for your user on the nasbox, or run with sudo locally.
ssh "$NASBOX" "sudo cat $REMOTE_CERT_DIR/fullchain.pem" > "$CERT_DIR/fullchain.pem"
ssh "$NASBOX" "sudo cat $REMOTE_CERT_DIR/privkey.pem"   > "$CERT_DIR/privkey.pem"

chmod 600 "$CERT_DIR/privkey.pem"
chmod 644 "$CERT_DIR/fullchain.pem"

# Show expiry as a sanity check
EXPIRY=$(openssl x509 -in "$CERT_DIR/fullchain.pem" -noout -enddate | cut -d= -f2)
echo "Certificates written to $CERT_DIR/"
echo "Expiry: $EXPIRY"
