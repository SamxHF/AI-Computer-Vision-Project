#!/usr/bin/env python3
"""Per-identity minimum cosine distances for selected MTCNN crops on the CD demo scene.

Mirrors people_identification_voting_simple.ipynb: ArcFace, MTCNN, same crop helper.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "open_data_set"
GALLERY_DIR = DATASET_ROOT / "photos_all_faces"
TEST_SCENE = DATASET_ROOT / "photos_all" / "cd_gp_0_eo_12.JPG"
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "mtcnn"


def list_images(folder: Path):
    out = []
    if not folder.is_dir():
        return out
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        out.extend(folder.glob(pat))
    return sorted(out)


def infer_identity(path: Path) -> str:
    return path.stem.split("_")[0].upper()


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a_norm = np.linalg.norm(a) + eps
    b_norm = np.linalg.norm(b) + eps
    return float(1.0 - np.dot(a, b) / (a_norm * b_norm))


def item_to_box_and_crop(item, scene_bgr, h0, w0):
    a = item["facial_area"]
    x = max(0, min(int(a["x"]), w0 - 1))
    y = max(0, min(int(a["y"]), h0 - 1))
    w = max(1, min(int(a["w"]), w0 - x))
    h = max(1, min(int(a["h"]), h0 - y))
    if w < 10 or h < 10:
        return None
    b = {"x": x, "y": y, "w": w, "h": h}
    face_arr = item.get("face")
    if face_arr is not None:
        fa = np.asarray(face_arr)
        if fa.dtype != np.uint8:
            fa = np.clip(fa * 255.0, 0, 255).astype(np.uint8)
        crop_rgb = fa[:, :, :3]
    else:
        crop_bgr = scene_bgr[b["y"] : b["y"] + b["h"], b["x"] : b["x"] + b["w"]]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return b, crop_rgb


def load_gallery_df() -> pd.DataFrame:
    paths = list_images(GALLERY_DIR)
    if not paths:
        raise FileNotFoundError(f"No images in {GALLERY_DIR}")
    rows = []
    embeddings = []
    for p in paths:
        rep = DeepFace.represent(
            img_path=str(p.resolve()),
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        embeddings.append(np.array(rep[0]["embedding"], dtype=np.float32))
        rows.append({"path": str(p.resolve()), "identity": infer_identity(p)})
    return pd.DataFrame(rows).assign(embedding=embeddings)


def scene_items_ok(scene_bgr: np.ndarray):
    h0, w0 = scene_bgr.shape[:2]
    raw_faces = DeepFace.extract_faces(
        img_path=scene_bgr,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )
    items_ok = []
    for item in raw_faces:
        pair = item_to_box_and_crop(item, scene_bgr, h0, w0)
        if pair is not None:
            items_ok.append(pair)
    return items_ok


def min_dist_per_identity(gallery_df: pd.DataFrame, probe: np.ndarray) -> dict[str, float]:
    out = {}
    for ident in gallery_df["identity"].unique():
        sub = gallery_df[gallery_df["identity"] == ident]
        out[str(ident)] = float(min(cosine_distance(probe, e) for e in sub["embedding"]))
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--faces",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Scene face indices (after MTCNN + min-size filter), same as notebook paired rows",
    )
    args = ap.parse_args()

    gallery_df = load_gallery_df()
    scene_bgr = cv2.imread(str(TEST_SCENE))
    if scene_bgr is None:
        raise FileNotFoundError(TEST_SCENE)

    items_ok = scene_items_ok(scene_bgr)
    print(f"Scene: {TEST_SCENE.name} | valid detections: {len(items_ok)}")

    for fi in args.faces:
        if fi < 0 or fi >= len(items_ok):
            print(f"\nSkip face #{fi}: out of range [0, {len(items_ok) - 1}]")
            continue
        _, crop_rgb = items_ok[fi]
        rep = DeepFace.represent(
            img_path=crop_rgb,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        emb = np.array(rep[0]["embedding"], dtype=np.float32)
        by_id = min_dist_per_identity(gallery_df, emb)
        ranked = sorted(by_id.items(), key=lambda x: x[1])

        print(f"\n=== Scene face #{fi} (MTCNN crop) ===")
        print("Per-identity minimum distance (1 - cos sim), best → worst:")
        for ident, d in ranked:
            mark = ""
            if ident in ("C", "D"):
                mark = "  ← expected CD token"
            print(f"  {ident}: {d:.6f}{mark}")
        print(f"Rank-1: {ranked[0][0]} @ {ranked[0][1]:.6f}")
        if "C" in by_id and "D" in by_id:
            print(f"C vs D gap (min_dist_C - min_dist_D): {by_id['C'] - by_id['D']:+.6f}")


if __name__ == "__main__":
    main()
