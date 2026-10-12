# `src/` audit checklist (Mac — not in Context sync)

Context sync copies `docs/`, `scripts/`, `internal/` only. **`src/` must be audited on `~/Projects/bayer_denoiser`.**

## Expected (current)

```
src/
  bayer_denoiser/          # import name; was bayer_rgb/
    train.py, infer.py, model.py, dataset.py, ...
    bayer_denoise_net.py
    (no sky/ — sky lives in scripts/sky/)
pyproject.toml             # package-dir bayer_denoiser = src
```

## Likely outdated if still present

| Path / pattern | Action |
|----------------|--------|
| `src/sky/` or `src/bayer_denoiser/sky/` | Remove after `git mv` to `scripts/sky/` ([sky-scripts-migration](../docs/internal/sky-scripts-migration.md)) |
| `src/bayer_rgb/` | Should be renamed to `bayer_denoiser` |
| `bayer_rgb` imports in `src/` | `git grep` → replace with `bayer_denoiser` |
| Duplicate sky CLI in package | Prefer `python scripts/sky/convert.py` only |

## Mac commands

```bash
cd ~/Projects/bayer_denoiser
find src -type f -name '*.py' | head -50
test -d src/sky && echo "OUTDATED: src/sky still exists"
test -d src/bayer_rgb && echo "OUTDATED: src/bayer_rgb still exists"
git grep -nI -e 'bayer_rgb' -e 'overfit' -- src ':!**/__pycache__'
ls -la src/bayer_denoiser 2>/dev/null || ls -la src/
```

## Deferred but not “delete”

- `GCPStillNet` / `--model gcp` — product deferred, code may remain.
- `demosaic` baseline — optional model path.
