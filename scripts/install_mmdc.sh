#!/usr/bin/env bash
# Install mermaid-cli globally. Idempotent.
set -euo pipefail

if ! command -v node >/dev/null 2>&1; then
    echo "node is required. Install with: brew install node  (or apt install nodejs npm)" >&2
    exit 1
fi

if command -v mmdc >/dev/null 2>&1; then
    echo "mmdc already installed at $(command -v mmdc)"
    exit 0
fi

echo "Installing @mermaid-js/mermaid-cli globally..."
npm install -g @mermaid-js/mermaid-cli

if ! command -v mmdc >/dev/null 2>&1; then
    echo "mmdc still not on PATH. You may need to add npm global bin to your shell:" >&2
    echo "  export PATH=\"\$(npm config get prefix)/bin:\$PATH\"" >&2
    exit 2
fi

echo "Installed: $(command -v mmdc)"
mmdc --version || true