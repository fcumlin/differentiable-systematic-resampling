#!/bin/bash
# Downloads the CharacterTrajectories data (UCI) to data/mixoutALL_shifted.mat.
# Usage (from the repository root): bash scripts/download_characters.sh
set -euo pipefail

URL=https://archive.ics.uci.edu/static/public/175/character+trajectories.zip
MD5=433cac77ca9bbb0aaac28951e146f519

mkdir -p data
curl -sSL -o data/character_trajectories.zip "${URL}"
unzip -o -q data/character_trajectories.zip mixoutALL_shifted.mat -d data
rm data/character_trajectories.zip
echo "${MD5}  data/mixoutALL_shifted.mat" | md5sum -c -
