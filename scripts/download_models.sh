#!/usr/bin/env bash
# Download the two local models with plain curl (bypasses occasional
# huggingface_hub stalls behind some middleboxes). Resumable with -C -.
#
# Usage: bash scripts/download_models.sh [mirror|hf]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/models"
HOST="${1:-mirror}"
if [ "$HOST" = "mirror" ]; then BASE="https://hf-mirror.com"; else BASE="https://huggingface.co"; fi

fetch() { # fetch <repo> <file> <outdir>
  mkdir -p "$3"
  curl -sL -C - --retry 5 --retry-delay 3 -o "$3/$2" "$BASE/$1/resolve/main/$2"
  echo "$2 -> $(wc -c < "$3/$2") bytes"
}

fetch BAAI/bge-reranker-base config.json        "$DEST/bge-reranker-base"
fetch BAAI/bge-reranker-base tokenizer.json     "$DEST/bge-reranker-base"
fetch BAAI/bge-reranker-base tokenizer_config.json "$DEST/bge-reranker-base"
fetch BAAI/bge-reranker-base special_tokens_map.json "$DEST/bge-reranker-base"
fetch BAAI/bge-reranker-base model.safetensors  "$DEST/bge-reranker-base"

echo "done. reranker will auto-load from $DEST/bge-reranker-base"
echo "(bge-small-zh-v1.5 embedder downloads via huggingface_hub on first use)"
