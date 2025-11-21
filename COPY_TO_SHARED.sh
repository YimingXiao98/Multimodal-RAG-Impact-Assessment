#!/bin/bash
# Script to copy essential files for annotation to shared folder

set -e  # Exit on error

DEST="/home/shared/YimingShared/DisasterRAG"

echo "📦 Copying files to $DEST..."
echo ""

# Create destination structure
echo "Creating directories..."
mkdir -p "$DEST/app"
mkdir -p "$DEST/scripts"
mkdir -p "$DEST/data/processed"
mkdir -p "$DEST/data/examples"
mkdir -p "$DEST/NOTES"

# Copy application code
echo "Copying application code..."
rsync -av --progress app/ "$DEST/app/"

# Copy annotation script
echo "Copying annotation script..."
cp scripts/annotate_retrieval_gt.py "$DEST/scripts/"

# Copy processed data
echo "Copying processed data (this may take a few minutes for tweets.parquet)..."
rsync -av --progress \
    data/processed/311.parquet \
    data/processed/tweets.parquet \
    data/processed/imagery_tiles.parquet \
    data/processed/gauges.parquet \
    data/processed/fema_kb.parquet \
    data/processed/claims.parquet \
    "$DEST/data/processed/"

# Copy documentation
echo "Copying documentation..."
cp NOTES/RETRIEVAL_GROUND_TRUTH_GUIDE.md "$DEST/NOTES/"
cp COWORKER_SETUP_GUIDE.md "$DEST/"
cp README.md "$DEST/"

# Copy configuration
echo "Copying configuration files..."
cp pyproject.toml "$DEST/"

# Create placeholder for output
touch "$DEST/data/examples/.gitkeep"

echo ""
echo "✅ Copy complete!"
echo ""
echo "Total size:"
du -sh "$DEST"
echo ""
echo "Next steps:"
echo "1. Your coworker should read: $DEST/COWORKER_SETUP_GUIDE.md"
echo "2. They should run: cd $DEST && conda create -n harvey-rag python=3.11 && conda activate harvey-rag && pip install -e ."
echo "3. Then start annotating: python scripts/annotate_retrieval_gt.py"
echo ""
echo "Output will be saved to: $DEST/data/examples/retrieval_gt.json"
