# Models

This folder stores the trained face recognition database generated after processing the registered face images.

## Contents

- `face_db.pkl` – Serialized database containing the average ArcFace embedding for each registered person.

## How It Is Generated

Run:

```bash
python train_test.py
```

This script:
1. Reads images from the `dataset/` folder.
2. Extracts ArcFace embeddings using DeepFace.
3. Computes the average embedding for each person.
4. Saves the database as `face_db.pkl` in this folder.

## Notes

- Do not edit `face_db.pkl` manually.
- If you add, remove, or update images in the `dataset/` folder, rerun `train_test.py` to regenerate the database.
- The application (`app_gui.py`) loads this file during real-time face recognition.
