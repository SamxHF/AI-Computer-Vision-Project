# Future ideas — face recognition & matching

Notes captured from project discussion for later experiments and report sections.

## When detection is OK but **matching is wrong**

Wrong identities usually mean a **recognition / decision** problem, not face detection.

### Common causes

1. **Visually similar people** (uniforms, pose, low resolution) → embeddings sit close together; nearest neighbor picks the wrong identity.
2. **Threshold too loose** → the system accepts a match when the best gallery hit is only slightly better than noise.
3. **Top-1 only** → no check that the winner is **clearly better** than the runner-up.
4. **Model / domain gap** → enrollment photos differ a lot from drone crops (sharp vs tiny, blur, lighting).
5. **Gallery imbalance or misleading single images** → one gallery photo can pull the wrong ID.

### Mitigations (typical order)

1. **Tighten the match threshold** — fewer false IDs, more “no match / unknown.”
2. **Margin / ambiguity reject** — require both `d1 < threshold` and `(d2 - d1) > margin` where `d1`, `d2` are best and second-best **distinct-identity** distances. If top two identities are tied, do not commit.
3. **Per-identity gallery fusion** — combine scores across all images of each person (see below).
4. **Stronger recognizer** — e.g. ArcFace-class models; InsightFace `Buffalo_L` if dependencies are available.
5. **Richer enrollment** — gallery images closer to drone conditions when possible.
6. **Calibration** — tune threshold on held-out same vs different pairs.

---

## Per-identity aggregation (gallery scoring)

Instead of trusting a single global nearest gallery **image**, you can **score each identity** using **all** of that person’s gallery photos, then pick the identity with the best **overall** evidence.

### 1. Best single match per identity (min over gallery)

For identity **A**: distance from probe to each gallery image of **A** → take the **minimum** (best match to A).  
Repeat for **B, C, …**  
Choose the identity with the **smallest** minimum.

**Note:** If the gallery is a flat list of labeled images, this is **equivalent** to standard **global nearest neighbor** (one argmin over all gallery rows). Changing **only** to min-per-ID without changing threshold or margin does **not** by itself fix wrong matches.

### 2. “Overall” support below threshold (voting / evidence)

For each identity, count how many gallery images have distance **below** a threshold (e.g. A: 4 under 0.15, B: 1 under 0.15).  
Pick the identity with the **most** strong hits, or require **at least K** hits before accepting.  
Otherwise output **no match**.

This **differs** from pure top-1 NN: one misleading close image can be outvoted by the rest of that person’s gallery disagreeing.

### 3. Average or top‑k average per identity

For each ID: **mean** distance to all gallery images, or **mean of the k smallest** distances (robust to outliers).  
Pick the ID with the lowest aggregate score; still apply threshold and/or margin.

---

## Pixel / preprocessing ideas (secondary to model + decision rules)

When faces are small or noisy, modest improvements sometimes help; they rarely fix a bad model or threshold.

- **Alignment and crop quality** — centered face with margin; use detector alignment where applicable.
- **Modest upscaling** (e.g. 1.25×–2×) with good interpolation; avoid huge upscale (blur, memory).
- **Lighting** — CLAHE on luminance, mild gamma; avoid over-sharpening.
- **Denoising** — light bilateral / NLM if noise dominates.

**Better imaging** (zoom, altitude, sensor) often beats heavy post-processing for tiny faces.

---

## “Tiny face” detectors

Specialized or well-tuned detectors for **very small** faces can help when you **also** give enough resolution (pyramid / upscale). They are **not** a substitute for pixels-on-face. Whether they beat MTCNN / RetinaFace on **your** data is empirical.

---

## References in this repo

- **Pipeline and objectives:** `PROJECT_OBJECTIVES.md`
- **How to run and repo layout:** `README.md`

When implementing any of the above, keep **one place** for `MATCH_THRESHOLD` (and optional `MARGIN`, `MIN_VOTES`) and use the same rules in visualization (e.g. section 6.5) and batch evaluation (section 8) so metrics match what you show.
