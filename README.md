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

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**FAISS notebook:** also install FAISS (CPU build is enough for this project):

```bash
pip install faiss-cpu
```

**DeepFace / TensorFlow:** If you see errors about `tf-keras`, install it (already listed in `requirements.txt`). Some optional models (e.g. certain InsightFace weights) need extra packages — the notebooks comment when that applies.

## How to run

1. Open **`people_identification_in_the_wild.ipynb`** (or the FAISS variant) in Jupyter, VS Code, or Cursor.
2. Run cells **from top to bottom** the first time so variables (`gallery_matrix`, `gallery_ids`, `gallery_paths` in the FAISS book, `fixed_boxes`, etc.) exist before later sections.
3. **Section 6.4** — fixed scene detector (MTCNN + NMS) and `fixed_boxes`.
4. **Section 6.5** — per-face gallery matching, plots, saved “all faces with IDs” image.
5. **Section 8** — batch loop over scenes, precision/recall style metrics vs filename tokens, **CSV** export (`batch_evaluation_results.csv` / `faiss_batch_evaluation_results.csv`).

Restart the kernel if you change model or gallery definitions mid-notebook.

## Pipeline (conceptual)

1. **Detect** faces in the scene (MTCNN via DeepFace; NMS removes duplicate boxes).
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

## License / data

If the dataset ships a license file (e.g. under `open_data_set/`), keep it with your copy of the data and cite it in your report.

## Further reading

- `PROJECT_OBJECTIVES.md` — project definition and suggested repositories.
- `FUTURE_IDEAS.md` — matching refinements and experiment ideas.
