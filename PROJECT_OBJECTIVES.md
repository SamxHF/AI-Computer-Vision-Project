# Project: People Identification in the Wild

## Overview
This project aims to develop an automated face recognition system for unconstrained environments ("in the wild").  
The main use case is identifying a target person in an open outdoor area using drone-captured imagery.

By completing this project, students will understand core concepts in:
- Computer vision
- Face detection
- Face recognition
- Identity matching in crowded scenes

## Scenario
- **Goal:** Recognize a specific person walking in an outdoor crowd.
- **Input:** Crowd image captured by a drone.
- **Output:** Position of the target person (bounding box) and identity (ID).

## General Algorithm
- **Input:** Query face image + classroom image or drone image
- **Output:** Localization and identification of the query face in the input image

### Pipeline
1. Detect all faces in the input image (`n` faces).
2. Crop all detected faces from the image (`n` cropped face images).
3. Extract face signatures from the `n` cropped faces (`s1, s2, ..., sn`), where each signature is a 1D feature vector.
4. Extract query signature (`qs`) from the query face.
5. Compute distances between `qs` and each detected face signature:  
   `d1 = distance(qs, s1), d2 = distance(qs, s2), ..., dn = distance(qs, sn)`.
6. Select the face with the minimum distance.
7. Mark the selected face with a green bounding box and assign the corresponding ID.

## Tasks

### Task 1: Data Collection
- Collect crowd video data (up to 20 people) in different configurations.
- Use online datasets when needed.

### Task 2: Implementation
- Build the full recognition pipeline:
  - Face detection
  - Face cropping
  - Feature/signature extraction
  - Query matching
  - Target localization and visualization

## Notes
- Main reference notebook:  
  [face_classi.ipynb (Google Drive)](https://drive.google.com/file/d/1v4wX3jooxX6EiBu42ZNIZYNL22TkEYo9/view?usp=sharing)

## Suggested Repositories for Implementation
The following repositories are recommended references for building and improving the pipeline:

- [InsightFace](https://github.com/deepinsight/insightface)  
  State-of-the-art face analysis toolkit (detection, alignment, and recognition).

- [DeepFace](https://github.com/serengil/deepface)  
  High-level Python library for face verification, recognition, and facial attribute analysis.

- [GhostFaceNets](https://github.com/HamadYA/GhostFaceNets)  
  Lightweight face recognition models suitable for resource-constrained or real-time systems.

- [MediaPipe Desktop Examples](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/examples/desktop)  
  Fast desktop demos and pipelines (including face detection) for real-time applications.

- [dlib Python Examples](https://github.com/davisking/dlib/tree/master/python_examples)  
  Classic Python examples for face detection and 128D face embeddings.
