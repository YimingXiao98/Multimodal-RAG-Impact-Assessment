# Prepare Repository for Public Push

Clean up the codebase and prepare it for a public git push. This command will:
1. Create a safety backup commit
2. Remove temporary/intermediate files
3. Archive old experiments and debug configs
4. Check for sensitive files (API keys, .env)
5. Verify the repo is public-ready
6. Commit the cleanup

## Instructions

Please perform the following cleanup steps in order:

### Step 1: Safety Backup
Create a backup commit so we can recover if needed:
```bash
git add -A && git commit -m "WIP: Backup before cleanup $(date +%Y%m%d-%H%M%S)" || echo "Nothing to commit"
```
Note the commit hash for recovery.

### Step 2: Run Cleanup Script
If `scripts/cleanup_codebase.sh` exists, run it. Otherwise, manually:
- Delete `*.intermediate.json` files in `data/experiments/`
- Delete `*.log` files in root and `data/experiments/`
- Delete `__pycache__/`, `.pytest_cache/`
- Delete LaTeX build artifacts (`paper/*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`, `*.fls`, `*.fdb_latexmk`)
- Archive old experiment files (exp0-13, validation_*) to `data/experiments/archive/`
- Archive debug configs to `config/archive/`

### Step 3: Clean Top-Level Files
Remove any stray log files from the project root:
```bash
rm -f *.log
```

### Step 4: Security Check
Verify NO sensitive files will be committed:
1. Check `.env` is gitignored: `git ls-files .env` (should be empty)
2. Check `.env.example` has only placeholders (no real keys)
3. Scan for hardcoded secrets: `grep -rn "sk-\|AIza" --include="*.py" app/ scripts/`
4. Check no large data files tracked: `git ls-files data/raw/ data/processed/` (should be empty)

### Step 5: Review Changes
Show what will be committed:
```bash
git status --short
```

### Step 6: Commit Cleanup
If everything looks good, commit:
```bash
git add -A && git commit -m "Clean codebase for public release

- Removed intermediate/temp files
- Archived old experiments and debug configs
- Updated .gitignore
- Verified no sensitive files tracked"
```

### Step 7: Final Verification
Confirm the repo is clean:
- Top-level should only have: `README.md`, `pyproject.toml`, `run.sh`, `demo_query.py`
- `data/experiments/` should only have final result files
- `config/` should only have production configs

### Recovery (if needed)
```bash
git reset --hard <backup-commit-hash>
```

## Output
Report:
1. Number of files deleted/archived
2. Security check results (PASS/FAIL)
3. Final commit hash
4. Any warnings or issues found

