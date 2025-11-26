#!/bin/bash
# Download JinaBERT model files for embedding
#
# Models are downloaded to crates/coppermind/assets/models/ where the
# Dioxus asset! macro expects them.

set -e

MODEL_DIR="crates/coppermind/assets/models"
MODEL_URL="https://huggingface.co/jinaai/jina-embeddings-v2-small-en/resolve/main"
MODEL_WEIGHTS="$MODEL_DIR/jina-bert.safetensors"
MODEL_TOKENIZER="$MODEL_DIR/jina-bert-tokenizer.json"

echo "Setting up model directory..."
mkdir -p "$MODEL_DIR"

echo "Downloading model files..."

if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Downloading model.safetensors (65MB)..."
    curl -C - --retry 10 --retry-delay 2 --retry-max-time 600 -L \
        "$MODEL_URL/model.safetensors" \
        -o "$MODEL_WEIGHTS"
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

if [ ! -f "$MODEL_TOKENIZER" ]; then
    echo "Downloading tokenizer.json (695KB)..."
    curl -C - --retry 5 -L \
        "$MODEL_URL/tokenizer.json" \
        -o "$MODEL_TOKENIZER"
    echo "✓ Tokenizer downloaded"
else
    echo "✓ Tokenizer already exists"
fi

echo ""
echo "All model files ready!"
ls -lh "$MODEL_DIR"
