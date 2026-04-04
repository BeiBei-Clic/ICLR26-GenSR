#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights"
CHECKPOINT_PATH="${WEIGHTS_DIR}/checkpoint.pth"
FILE_ID="1TbcRSzO3rGQBuJIPN5P6fQpYYEv4__K4"
DOWNLOAD_URL="https://drive.usercontent.google.com/download?id=${FILE_ID}&export=download&confirm=t"

mkdir -p "${WEIGHTS_DIR}"

curl -L "${DOWNLOAD_URL}" -o "${CHECKPOINT_PATH}"

echo "checkpoint saved to ${CHECKPOINT_PATH}"
