#!/bin/bash
# Download JinaBert model files for embedding

set -e

MODEL_DIR="assets/models"
MODEL_URL="https://huggingface.co/jinaai/jina-embeddings-v2-small-en/resolve/main"
MODEL_WEIGHTS="$MODEL_DIR/jina-bert.safetensors"
MODEL_TOKENIZER="$MODEL_DIR/jina-bert-tokenizer.json"

echo "Setting up model directory..."
mkdir -p "$MODEL_DIR"

# Migrate legacy filenames if present
if [ -f "$MODEL_DIR/model.safetensors" ] && [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Renaming legacy model.safetensors → jina-bert.safetensors"
    mv "$MODEL_DIR/model.safetensors" "$MODEL_WEIGHTS"
fi

if [ -f "$MODEL_DIR/tokenizer.json" ] && [ ! -f "$MODEL_TOKENIZER" ]; then
    echo "Renaming legacy tokenizer.json → jina-bert-tokenizer.json"
    mv "$MODEL_DIR/tokenizer.json" "$MODEL_TOKENIZER"
fi

echo "Downloading model files..."

# Download model.safetensors if not exists
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "Downloading model.safetensors (262MB)..."
    curl -C - --retry 10 --retry-delay 2 --retry-max-time 600 -L \
        "$MODEL_URL/model.safetensors" \
        -o "$MODEL_WEIGHTS"
    echo "✓ Model downloaded"
else
    echo "✓ Model already exists"
fi

# Download tokenizer.json if not exists
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
