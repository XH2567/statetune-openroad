#!/bin/bash
# Install the OpenROAD-flow-scripts (ORFS) toolchain WITHOUT root or a docker daemon:
#   1) download `crane` (single static binary) to $ORFS_HOME/bin
#   2) pull the pinned openroad/orfs image and flatten it into $ORFS_HOME/rootfs (~4 GB)
#   3) smoke-test openroad / yosys through tools/orfs-sandbox (bubblewrap)
#
# Usage:  ORFS_HOME=/path/with/space/orfs tools/setup_orfs.sh
# Env:    ORFS_REGISTRY  image registry/mirror (default docker.1ms.run; use docker.io outside CN,
#                        or another docker hub mirror that allows openroad/orfs)
#         HTTPS_PROXY    honoured by crane if set
set -euo pipefail

: "${ORFS_HOME:?set ORFS_HOME to the install directory}"
REGISTRY="${ORFS_REGISTRY:-docker.1ms.run}"
# The image used for all reported results (ORFS 26Q4-180-g5110a7c29, built 2026-10-21).
# Pin by digest so everyone gets byte-identical tools even after the tag/latest moves.
IMAGE_TAG="26Q4-180-g5110a7c29"
IMAGE_DIGEST="sha256:46b6800cc65e936850a5993300849961484656585212edebb9cbcc953fb2b4f3"
CRANE_VERSION="v0.22.1"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v bwrap >/dev/null || { echo "bubblewrap (bwrap) is required: apt install bubblewrap, or ask your admin" >&2; exit 1; }
bwrap --ro-bind / / --dev /dev true 2>/dev/null || { echo "bwrap cannot create a sandbox on this host (user namespaces disabled?)" >&2; exit 1; }

mkdir -p "$ORFS_HOME/bin" "$ORFS_HOME/rootfs"
export TMPDIR="${TMPDIR:-$ORFS_HOME/tmp}"; mkdir -p "$TMPDIR"

if [ ! -x "$ORFS_HOME/bin/crane" ]; then
  echo "[1/3] downloading crane $CRANE_VERSION"
  curl -fsSL "https://github.com/google/go-containerregistry/releases/download/$CRANE_VERSION/go-containerregistry_Linux_x86_64.tar.gz" \
    | tar -xz -C "$ORFS_HOME/bin" crane
fi

if [ -d "$ORFS_HOME/rootfs/OpenROAD-flow-scripts" ]; then
  echo "[2/3] $ORFS_HOME/rootfs already populated, skipping pull (delete it to re-pull)"
else
  echo "[2/3] pulling $REGISTRY/openroad/orfs:$IMAGE_TAG@$IMAGE_DIGEST (~1.4 GB download, ~4 GB on disk)"
  "$ORFS_HOME/bin/crane" export --platform linux/amd64 "$REGISTRY/openroad/orfs@$IMAGE_DIGEST" - \
    | tar -x -C "$ORFS_HOME/rootfs" --no-same-owner --no-same-permissions
fi

echo "[3/3] smoke test"
export ORFS_HOME
"$REPO/tools/orfs-sandbox" openroad -version >/dev/null && echo "  openroad: ok"
"$REPO/tools/orfs-sandbox" yosys -V
echo "done. Put  export ORFS_HOME=$ORFS_HOME  in your .env"
