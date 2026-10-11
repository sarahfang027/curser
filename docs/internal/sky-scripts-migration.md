---
cursor:
  subagentId: "bc-014eb380-6b56-57c2-93ac-007e8bd71239"
---

# Sky data prep: `src/sky/` → `scripts/sky/` (Mac canonical)

**Repo:** `~/Projects/bayer_rgb`  
**Scope:** `.sky` (csky) → `.npy` conversion only — not part of the installable training package (`bayer_rgb` train/infer/preview).  
**Worker status (2026-09-29):** No Mac worker; no `bayer_rgb` checkout on cloud VM; GitHub remote not accessible from worker — **apply these steps manually on the Mac**.

Context sync script (`scripts/sync_context_to_repo.sh`) copies **Context → repo** docs/scripts only; it does **not** move Python package layout. **No sync-script change** for this migration.

---

## 1. Preconditions

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
git status   # clean or dedicated branch
git checkout -b chore/sky-scripts-folder   # optional
```

Confirm current layout (expected from project docs):

| Path | Role |
|------|------|
| `src/sky/io.py` | csky decode, row deinterleave, normalize |
| `src/sky/convert.py` | CLI (`argparse`), calls `io` |
| `src/sky/__init__.py` | optional package marker |
| `src/bayer_rgb/sky/` **or** `bayer_rgb/sky/` | thin package so `python -m bayer_rgb.sky` works |

If your tree differs, grep for `bayer_rgb.sky` and `src/sky` before moving.

---

## 2. Move files

```bash
cd ~/Projects/bayer_rgb
mkdir -p scripts/sky
git mv src/sky/io.py scripts/sky/io.py
git mv src/sky/convert.py scripts/sky/convert.py
# Keep __init__.py only if convert/io use relative imports or you want scripts/sky as a local package:
test -f src/sky/__init__.py && git mv src/sky/__init__.py scripts/sky/__init__.py
rmdir src/sky 2>/dev/null || true
```

**Imports inside `scripts/sky/`:** use sibling imports only, e.g. `from io import ...` or `from .io import ...` with `__init__.py` present. Do **not** import `bayer_rgb` from sky scripts (keeps data prep decoupled).

---

## 3. Preferred CLI (pick one — recommended below)

### Recommended: direct script entry (no `PYTHONPATH`, no extra package)

Run from **repo root**:

```bash
cd ~/Projects/bayer_rgb
python scripts/sky/convert.py \
  --input "/path/to/clip.sky" \
  --output "data/samples/clip.npy" \
  --write-json "data/samples/clip.sky.json"
```

Ensure `convert.py` ends with:

```python
if __name__ == "__main__":
    main()  # or raise SystemExit(main())
```

**Why:** Sky is data prep, not `pip install -e .` surface area; matches other repo `scripts/*.py` patterns.

### Optional: `python -m` (only if you want module-style invocation)

Add `scripts/sky/__main__.py`:

```python
from convert import main

if __name__ == "__main__":
    raise SystemExit(main())
```

Then document:

```bash
cd ~/Projects/bayer_rgb
PYTHONPATH=scripts/sky python -m sky \
  --input ... --output ...
```

(`PYTHONPATH=scripts/sky` makes the folder importable as top-level package name `sky`.) **Do not** add `scripts` to `bayer_rgb` package discovery.

### Not recommended for this repo

- `console_scripts` entry `bayer-sky` in `pyproject.toml` — extra install coupling for a one-off converter unless you strongly want `bayer-sky` on PATH after `pip install -e .`.

---

## 4. `pyproject.toml` / packaging

After the move, **remove sky from the installable package**:

1. Open `pyproject.toml` (or legacy layout: no file — then only delete shim modules).
2. If `[tool.setuptools.packages]` / `packages` lists `bayer_rgb.sky` or discovers `src/bayer_rgb/sky`, **remove** that package.
3. If `package-dir` mapped `bayer_rgb.sky` → `src/sky`, **delete that mapping**.
4. Rerun:

```bash
pip install -e .
rm -rf bayer_rgb.egg-info src/*.egg-info  # if stale
pip install -e .
```

**Training CLIs unchanged:** `python -m bayer_rgb.train`, `preview`, `infer`, etc.

---

## 5. Remove or thin `bayer_rgb.sky` under `src/`

If `src/bayer_rgb/sky/` (or `bayer_rgb/sky/`) only re-exported `src/sky` or hosted `__main__.py` for `-m bayer_rgb.sky`:

**Option A — delete (clean break):** remove the whole `bayer_rgb/sky` tree after docs are updated.

**Option B — one-release deprecation shim (optional):**

Keep `src/bayer_rgb/sky/__main__.py` (and minimal `__init__.py`) for **one release**:

```python
# src/bayer_rgb/sky/__main__.py
import sys
import warnings
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]  # adjust depth: repo root
_CONVERT = _REPO / "scripts" / "sky" / "convert.py"

def main() -> int:
    warnings.warn(
        "python -m bayer_rgb.sky is deprecated; use: "
        "python scripts/sky/convert.py (from repo root)",
        DeprecationWarning,
        stacklevel=1,
    )
    print(
        "Deprecated: use from repo root:\n"
        "  python scripts/sky/convert.py --input ... --output ...",
        file=sys.stderr,
    )
    # Delegate: re-exec convert.py with same argv tail
    import runpy
    sys.argv[0] = str(_CONVERT)
    runpy.run_path(str(_CONVERT), run_name="__main__")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Adjust `_REPO` if your `src` layout differs (`parents[2]` vs `[3]`). Remove shim in the following release.

---

## 6. Verification

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate

python scripts/sky/convert.py \
  --input "/path/to/sample.sky" \
  --output /tmp/clip.npy \
  --write-json /tmp/clip.sky.json

python -c "import numpy as np; a=np.load('/tmp/clip.npy'); print(a.shape, a.dtype)"
# expect float16 H×W per float16-pipeline-standard

python -m bayer_rgb.preview /tmp/clip.npy --tile-size 512 --center
```

If deprecation shim kept:

```bash
python -W default::DeprecationWarning -m bayer_rgb.sky --help
```

---

## 7. Grep checklist (repo + Context archive)

Run from repo root after edits:

```bash
cd ~/Projects/bayer_rgb
rg -n 'bayer_rgb\.sky|src/sky|python -m bayer_rgb\.sky' --glob '!*.egg-info' .
```

### Strings to replace (from archived `internal/archived/context-docs-2026-09-29/sky-format-conversion.md`)

| Find | Replace with |
|------|----------------|
| `python -m bayer_rgb.sky` | `python scripts/sky/convert.py` |
| `Implementation: **\`src/sky/convert.py\`** (+ **\`src/sky/io.py\`**).` | `Implementation: **\`scripts/sky/convert.py\`** (+ **\`scripts/sky/io.py\`**).` |
| `Run from the **repo root** (\`~/Projects/bayer_rgb\`), not from \`src/sky/\`` | `Run from the **repo root** (\`~/Projects/bayer_rgb\`), not from \`scripts/sky/\`` |
| `Code: \`src/sky/\` (\`io.py\`, \`convert.py\`). Import name remains **\`bayer_rgb\`**.` | `Code: \`scripts/sky/\` (\`io.py\`, \`convert.py\`). Training package remains **\`bayer_rgb\`**.` |
| `pip install -e .    # once: makes \`bayer_rgb\` importable` (sky section) | Keep for **train/preview** only; sky CLI does not require `-m bayer_rgb.sky` |
| `python -m bayer_rgb.sky --input clip.sky --output /tmp/t.npy` (float16 checklist) | `python scripts/sky/convert.py --input clip.sky --output /tmp/t.npy` |

Also search:

- `docs/sky-format-conversion.md` (canonical after Mac doc sync)
- `docs/internal/**`, `internal/archived/**` copies
- `internal/float16-pipeline-implementation.md` (Context: `src/sky/convert.py` paths)
- README / runbooks mentioning sky module

**Context archive path (reference only):**  
`internal/archived/context-docs-2026-09-29/sky-format-conversion.md`

---

## 8. Patch for repo `docs/sky-format-conversion.md` (after sync)

Apply on Mac (or edit in Context archive and re-sync). Key sections:

### Setup (revised)

```markdown
## Setup

Run from the **repo root** (`~/Projects/bayer_rgb`).

```bash
cd ~/Projects/bayer_rgb
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .    # train / preview / infer — not required for sky convert alone
```
```

### Command (revised)

```markdown
## Command

Use **`python scripts/sky/convert.py`** from repo root. Implementation: **`scripts/sky/convert.py`** (+ **`scripts/sky/io.py`**).

```bash
cd ~/Projects/bayer_rgb
python scripts/sky/convert.py \
  --input "/path/to/clip.sky" \
  --output "data/samples/clip.npy" \
  --write-json "data/samples/clip.sky.json"
```
```

### Verify — preview unchanged

`python -m bayer_rgb.preview data/samples/clip.npy ...` stays as-is (core package).

### Footer (revised)

```markdown
Code: `scripts/sky/` (`io.py`, `convert.py`). Training/import package: **`bayer_rgb`**.
```

Full pre-migration reference doc is preserved in Context:  
[sky-format-conversion.md (archived 2026-09-29)](archived/context-docs-2026-09-29/sky-format-conversion.md).

---

## 9. Git commit (Mac)

```bash
git add scripts/sky src/sky pyproject.toml docs/sky-format-conversion.md
git status
git commit -m "chore: move sky conversion from src/sky to scripts/sky"
```

---

## 10. Worker / cloud note

Cloud agent could not clone `bayer_rgb` (remote not found / private). **No PR from worker.** Implement on Mac canonical checkout; push when ready.
