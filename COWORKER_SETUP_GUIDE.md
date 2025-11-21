# Retrieval Ground Truth Annotation Guide

Welcome! This guide will help you set up and use the annotation tool to create ground truth data for our Hurricane Harvey RAG system.

## What You're Doing

You'll be reviewing retrieval results (imagery, tweets, 311 calls) and marking which documents are **actually relevant** for specific queries. This creates an "answer key" we can use to evaluate the system's performance.

## Setup (5 minutes)

### 1. Navigate to the Project
```bash
cd /home/shared/YimingShared/DisasterRAG
```

### 2. Create a Conda Environment
```bash
# Create environment
conda create -n harvey-rag python=3.11 -y

# Activate it
conda activate harvey-rag

# Install dependencies
pip install -e .
```

### 3. Verify Setup
```bash
# Test that the annotation tool loads
python scripts/annotate_retrieval_gt.py --help
```

You should see help text with usage instructions.

## Running the Annotation Tool

### Start Annotation
```bash
conda activate harvey-rag
python scripts/annotate_retrieval_gt.py
```

### What Happens Next

The tool will:
1. **Show you a query** (e.g., "ZIP 77002, Aug 26-30, 2017")
2. **Run retrieval** to find relevant documents
3. **Display results** in categories:
   - Imagery tiles
   - Tweets
   - 311 calls
4. **Ask you to judge** which are relevant

### How to Annotate

For each category, you'll see numbered items:

```
Found 8 imagery tiles:
  [1] 9-10-2017_Ortho_ColorBalance - 2017-09-10
      URI: /data/imagery/...
  [2] 9-11-2017_Ortho_ColorBalance - 2017-09-11
      URI: /data/imagery/...
  ...

Enter the numbers of relevant imagery tiles (comma-separated, or 'all'/'none'):
>
```

**Your options:**
- Type numbers: `1,2,5` (items 1, 2, and 5 are relevant)
- Type `all` (all items are relevant)
- Type `none` or press Enter (no items are relevant)

### Judging Relevance

**For Imagery:**
✅ **RELEVANT** if it shows:
- Visible flooding or water damage
- Structural damage from Harvey
- Taken during or shortly after the query time period
- In the query ZIP code or nearby

❌ **NOT RELEVANT** if:
- Clear/undamaged imagery
- Wrong time period (too early or too late)
- Too far from query location

**For Tweets/311 Calls:**
✅ **RELEVANT** if it:
- Explicitly mentions flooding, damage, or Harvey
- Is from the query ZIP code or area
- Was posted during the query time window

❌ **NOT RELEVANT** if:
- Generic news or non-specific content
- Different location
- Wrong time period

### Tips for Consistency

1. **When in doubt, include it** - Better to have false positives than miss relevant docs
2. **Be consistent** - Use the same criteria for every query
3. **Don't overthink** - Your first judgment is usually correct
4. **Take breaks** - Annotation can be tiring; do 5-10 queries at a time

## Saving Your Work

The tool auto-saves after each query to:
```
data/examples/retrieval_gt.json
```

You can press **Ctrl+C** anytime to exit - your progress is saved!

## Resuming Later

If you exit and come back:
```bash
conda activate harvey-rag
python scripts/annotate_retrieval_gt.py
```

The tool will **skip already-annotated queries** and continue from where you left off.

## Example Session

```
Starting annotation for 2 queries...
(Press Ctrl+C to save and exit early)

[1/2] Annotating 77002_2017-08-26_2017-08-30

================================================================================
QUERY: ZIP 77002, 2017-08-26 to 2017-08-30
================================================================================

Running retrieval...

Found 8 imagery tiles:
  [1] 9-10-2017_Ortho_ColorBalance - 2017-09-10
  [2] 9-11-2017_Ortho_ColorBalance - 2017-09-11
  ...

Enter the numbers of relevant imagery tiles (comma-separated, or 'all'/'none'):
> 1,2

Found 15 tweets:
  [1] Tweet 914278414869245952
      Houston downtown is completely flooded. Main St underwater...
  ...

Enter the numbers of relevant tweets (comma-separated, or 'all'/'none'):
> 1,3,5

Found 10 311 calls:
  [1] Call 12170210-101002743775
      Type: Flooded Street
      Street flooding on Louisiana St...
  ...

Enter the numbers of relevant 311 calls (comma-separated, or 'all'/'none'):
> 1,2

✅ Annotated: 2 imagery, 5 text docs
💾 Saved to data/examples/retrieval_gt.json
```

## How Many Queries?

**Goal:** Annotate **10-20 queries** minimum
- Each query takes ~3-5 minutes
- Total time: ~30-60 minutes

## Questions?

If you run into issues:
1. Check that you're in the `harvey-rag` conda environment
2. Verify files exist in `data/processed/`
3. Contact me if you get errors

## What Happens Next?

Once you finish annotating, send me the file:
```
data/examples/retrieval_gt.json
```

I'll use it to evaluate the retrieval system and calculate metrics like:
- **Precision**: % of retrieved docs that are relevant
- **Recall**: % of relevant docs that were retrieved
- **F1 Score**: Overall retrieval quality

Thank you for your help! 🙏
