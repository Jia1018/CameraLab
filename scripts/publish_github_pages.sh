#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 git@github.com:USER/camera_motion_disentangle.git"
  echo "   or: $0 https://github.com/USER/camera_motion_disentangle.git"
  exit 2
fi

repo_url="$1"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$project_root"

if [[ ! -d .git ]]; then
  git init
fi

git remote remove origin 2>/dev/null || true
git remote add origin "$repo_url"

git add README.md scripts site .gitignore
git commit -m "Add camera motion disentanglement result site" || true
git branch -M main
git push -u origin main

echo
echo "Pushed to $repo_url"
echo "Enable GitHub Pages in repo Settings -> Pages -> Deploy from branch -> main /site."
