#!/bin/bash
# Download JinaBert model files for embedding

set -e

MODEL_DIR="assets/models"
MODEL_URL="https://huggingface.co/jinaai/jina-embeddings-v2-small-en/resolve/main"

echo "Setting up model directory..."
mkdir -p "$MODEL_DIR"

echo "Downloading model files..."

# Download model.safetensors if not exists
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Downloading model.safetensors (262MB)..."
    curl -C - --retry 10 --retry-delay 2 --retry-max-time 600 -L \
        "$MODEL_URL/model.safetensors" \
        -o "$MODEL_DIR/model.safetensors"
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

# Download tokenizer.json if not exists
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "Downloading tokenizer.json (695KB)..."
    curl -C - --retry 5 -L \
        "$MODEL_URL/tokenizer.json" \
        -o "$MODEL_DIR/tokenizer.json"
    echo "✓ Tokenizer downloaded"
else
    echo "✓ Tokenizer already exists"
fi

echo ""
echo "All model files ready!"
ls -lh "$MODEL_DIR"
