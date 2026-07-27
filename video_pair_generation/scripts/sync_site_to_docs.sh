#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

rm -rf docs
mkdir -p docs
tar -C site --exclude='*/frames' -cf - . | tar -C docs -xf -

echo "Synced site/ to docs/ for GitHub Pages."
