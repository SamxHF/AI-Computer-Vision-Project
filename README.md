# People Identification in the Wild

Face-based **identification** for drone-style crowd images: detect faces in a scene, compare them to a **gallery** of known people, and assign **IDs** (with optional evaluation against filename-based ground truth).

## What this project does

- **Input:** Images from your dataset under `open_data_set/` (not committed to git — see below).
- **Output:** Predicted identity per detected face, annotated images on disk, and CSV summaries from batch evaluation.

The teaching scenario is: **crowd / outdoor image** → **bounding boxes** + **who each face is** (gallery match), aligned with the course “identification in the wild” objective.

## Repository layout

| Item | Purpose |
|------|--------|
| `people_identification_in_the_wild.ipynb` | Main pipeline: DeepFace (MTCNN detection, ArcFace-style embeddings), gallery matching, per-scene evaluation, CSV export. |
| `people_identification_in_the_wild_faiss.ipynb` | Same ideas; gallery nearest-neighbor via **FAISS** (`IndexFlatIP` on L2-normalized vectors) for speed on large galleries. |
| `PROJECT_OBJECTIVES.md` | Formal objectives, scenario, suggested GitHub repos (InsightFace, DeepFace, etc.). |
| `FUTURE_IDEAS.md` | Optional improvements: per-identity aggregation, margins, thresholds, preprocessing — for reports and later experiments. |
| `requirements.txt` | Python dependencies for the main notebook. |
| `face_classi.ipynb` | Reference / starter material from the course. |

## Dataset

- Expected path: **`open_data_set/`** at the project root (created when you unzip or copy the course dataset).
- `open_data_set/` is listed in **`.gitignore`** so large binaries are not pushed to GitHub.
- The notebooks build a metadata table from folder structure: **gallery**, **query_pool**, **scene_pool** (`photos_all`), etc.

If cells fail with “file not found,” confirm the dataset exists locally and paths in the notebook match your tree.

## Setup

Use a **supported Python** for TensorFlow (see [tensorflow.org/install](https://www.tensorflow.org/install)) — typically **3.9–3.12**. **Python 3.13** with DeepFace/TensorFlow often **crashes the Jupyter kernel** on import; prefer **3.11** or **3.10**.

```bash
# Example: ensure 3.11 (e.g. Homebrew: brew install python@3.11)
cd /path/to/AI-Computer-Vision-Project
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

**VS Code / Cursor:** use **Python: Select Interpreter** and pick **`.venv/bin/python`** for this folder. Do **not** rely on **Anaconda `base`** if the kernel dies on the first cell — `base` is often **Python 3.13** and mixes conda NumPy/BLAS with pip TensorFlow, which can **segfault** on `import tensorflow` or `from deepface import DeepFace`.

**FAISS notebook:** also install FAISS (CPU build is enough for this project):

```bash
pip install faiss-cpu
```

**DeepFace / TensorFlow:** If you see errors about `tf-keras`, install it (already listed in `requirements.txt`). Some optional models (e.g. certain InsightFace weights) need extra packages — the notebooks comment when that applies.

## How to run

1. Open **`people_identification_in_the_wild.ipynb`**, the **Aggregate** notebook (`people_identification_in_the_wild_Aggreggate.ipynb`, **mean** cosine distance per identity, argmin + threshold), the **Voting** notebook (`people_identification_in_the_wild_voting.ipynb`, min vote floor then pick highest count — see `FUTURE_IDEAS.md`), the **Ranking** notebook (`people_identification_in_the_wild_Ranking.ipynb`, margin between top-two identities), or the FAISS variant in Jupyter, VS Code, or Cursor.
2. Run cells **from top to bottom** the first time so variables (`gallery_matrix`, `gallery_ids`, `gallery_paths` in the FAISS book, `fixed_boxes`, etc.) exist before later sections. In the Aggregate/Voting/Ranking books, run **section 6.5.0** before 6.5 and section 8 so matching thresholds and helpers are defined.
3. **Section 6.4** — fixed scene detector (**MTCNN**; raw boxes, no extra NMS) and `fixed_boxes`.
4. **Section 6.5** — per-face gallery matching (voting or margin rule in the dedicated notebooks), plots, saved “all faces with IDs” image.
5. **Section 8** — batch loop over scenes, precision/recall style metrics vs filename tokens, **CSV** export (`batch_evaluation_results.csv` / `faiss_batch_evaluation_results.csv`).

Restart the kernel if you change model or gallery definitions mid-notebook.

## Pipeline (conceptual)

1. **Detect** faces in the scene (MTCNN via DeepFace; Aggregate/Voting/Ranking notebooks keep raw detector boxes without extra NMS).
2. **Crop** each face region.
3. **Embed** each crop with a recognition model (default **ArcFace** in DeepFace).
4. **Match** each embedding to the **gallery** (nearest neighbor by cosine distance, or FAISS inner product on normalized vectors).
5. **Decide** accept/reject using a distance **threshold** (and optionally stricter rules — see `FUTURE_IDEAS.md`).

Earlier cells may still build a **query** embedding for demos or timing; **main ID assignment in 6.5 and batch eval is gallery-based** (not query-to-scene localization).

## Outputs you may see

- `output_all_detected_faces_with_ids.jpg` / `faiss_output_all_detected_faces_with_ids.jpg` — scene with boxes and labels.
- `batch_evaluation_results.csv` / `faiss_batch_evaluation_results.csv` — per-scene / per-face evaluation rows.

Paths are written relative to the notebook’s `PROJECT_ROOT` (repo root).

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Missing `open_data_set` | Add the dataset locally; do not rely on git for it. |
| `gallery_paths` missing (FAISS 6.5) | Re-run the **gallery embedding** cell that builds `gallery_matrix` / `gallery_ids` **and** `gallery_paths` in one pass. |
| Kernel crash on huge images | Notebooks cap very large upscales; prefer MTCNN without extreme full-frame upscale. |
| Wrong IDs, detection OK | Tighten threshold; consider margin between 1st and 2nd identity — see `FUTURE_IDEAS.md`. |

### Kernel dies immediately, or segfault on `import tensorflow` / `DeepFace`

**Symptoms:** Jupyter shows *“The kernel died”* or *“kernel process died”* right after starting, often on the **first code cell** (`from deepface import DeepFace`). In a terminal, `python -c "import tensorflow as tf"` exits with **segmentation fault** and no Python traceback.

**Typical causes:**

- Notebook kernel is **Anaconda `base`** with **Python 3.13** — TensorFlow may not support that version cleanly yet, or wheels may be incompatible.
- **Mixed stack:** conda-managed NumPy/BLAS plus **pip-installed TensorFlow** in the same environment can trigger **native crashes** on import.

**Fix (recommended):**

1. Create a **fresh venv** with **Python 3.11** (or 3.10), not 3.13:

   ```bash
   cd /path/to/AI-Computer-Vision-Project
   rm -rf .venv
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. Verify outside the notebook:

   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   python -c "from deepface import DeepFace; print('deepface ok')"
   ```

3. In **VS Code / Cursor**, **Python: Select Interpreter** → choose **`./.venv/bin/python`** (not `/opt/anaconda3/bin/python` or “base”).

4. Re-open the notebook and confirm the kernel label shows your **3.11** venv, not **base (Python 3.13.x)**.

**Optional:** Jupyter may log a **widget / unpkg.com** CDN error; that is unrelated to TensorFlow and does not fix kernel crashes by itself.

### `NotFoundError: libmetal_plugin.dylib` / `_pywrap_tensorflow_internal.so` (macOS)

**Symptoms:** Importing `DeepFace` or `tensorflow` raises `NotFoundError: dlopen(... tensorflow-plugins/libmetal_plugin.dylib ...)` and mentions **`~/Library/Python/3.9/...`** (or similar) even though you use a **project `.venv`**.

**Cause:** A **second TensorFlow / Metal plugin** was installed with **`pip install --user`** (user site-packages). TensorFlow then loads **Metal plugins from that folder**, but they no longer match the TensorFlow build inside `.venv`, so a core `.so` is “not found”.

**Fix (pick one or combine):**

1. **Ignore user site** (quick check from a terminal — not required if you do step 2):

   ```bash
   cd /path/to/AI-Computer-Vision-Project
   source .venv/bin/activate
   PYTHONNOUSERSITE=1 python -c "from deepface import DeepFace; print('ok')"
   ```

   If that works, the problem is definitely the user `~/Library/Python/...` install; step 2 fixes it so you do not need this flag for normal use.

2. **Remove user-level TensorFlow** (recommended — fixes notebooks without any extra env files):

   ```bash
   python3.9 -m pip uninstall tensorflow tensorflow-macos tensorflow-metal tensorflow-metal-plugin tf-keras keras -y
   ```

   Repeat until none are listed; if needed, remove leftover folders:

   `~/Library/Python/3.9/lib/python/site-packages/tensorflow*`  
   `~/Library/Python/3.9/lib/python/site-packages/tensorflow-plugins`

   Then re-activate `.venv` and run `pip install -r requirements.txt` again.

**Going forward:** Prefer **only** installing into `.venv` (`pip` after `source .venv/bin/activate`); avoid **`pip install --user`** for TensorFlow on the same Python version.

## License / data

If the dataset ships a license file (e.g. under `open_data_set/`), keep it with your copy of the data and cite it in your report.

## Further reading

- `PROJECT_OBJECTIVES.md` — project definition and suggested repositories.
- `FUTURE_IDEAS.md` — matching refinements and experiment ideas.
