import os
import pickle
import numpy as np
from deepface import DeepFace

DATASET_DIR = "dataset"
MODEL_NAME = "ArcFace"

database = {}

print("[INFO] Building embedding database...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            emb = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend="opencv",
                enforce_detection=False
            )[0]["embedding"]

            embeddings.append(emb)

        except:
            pass

    if len(embeddings) > 0:
        # store average embedding (very important)
        database[person] = np.mean(embeddings, axis=0)
        print(f"[OK] {person} -> {len(embeddings)} samples")

with open("models/face_db.pkl", "wb") as f:
    pickle.dump(database, f)

print("[SUCCESS] Database created!")