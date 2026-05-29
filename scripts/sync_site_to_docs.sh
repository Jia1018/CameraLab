#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

rm -rf docs
mkdir -p docs
cp -a site/. docs/
find docs/assets/runs -type d -name frames -prune -exec rm -rf {} +

echo "Synced site/ to docs/ for GitHub Pages."
