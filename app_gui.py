import cv2
import pickle
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from deepface import DeepFace

# Load database
with open("models/face_db.pkl", "rb") as f:
    database = pickle.load(f)

MODEL_NAME = "ArcFace"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

root = tk.Tk()
root.title("Face Recognition")

label = tk.Label(root)
label.pack()

frame_count = 0
PROCESS_EVERY = 5
last_results = []

# KEY PARAMETER (tune this)
DIST_THRESHOLD = 0.65


def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def update():
    global frame_count, last_results

    ret, frame = cap.read()
    if not ret:
        label.after(10, update)
        return

    if frame_count % PROCESS_EVERY == 0:
        new_results = []

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False
            )

            for face in faces:
                x = face["facial_area"]["x"]
                y = face["facial_area"]["y"]
                w = face["facial_area"]["w"]
                h = face["facial_area"]["h"]

                face_img = frame[y:y+h, x:x+w]

                if face_img.size == 0:
                    continue

                face_img = cv2.resize(face_img, (224, 224))

                emb = DeepFace.represent(
                    img_path=face_img,
                    model_name=MODEL_NAME,
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]

                # Compare with database
                best_name = "Unknown"
                best_dist = 999

                for name, db_emb in database.items():
                    dist = cosine_distance(emb, db_emb)

                    if dist < best_dist:
                        best_dist = dist
                        best_name = name

                # FINAL DECISION
                if best_dist > DIST_THRESHOLD:
                    best_name = "Unknown"

                new_results.append((x, y, w, h, best_name, best_dist))

            if len(new_results) > 0:
                last_results = new_results

        except Exception as e:
            print("Error:", e)

    # Draw
    for (x, y, w, h, name, dist) in last_results:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        text = f"{name} ({dist:.2f})"
        cv2.putText(frame, text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)

    frame_count += 1

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    label.after(10, update)


update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()