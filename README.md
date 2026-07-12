# Real-Time Face Recognition System

A Python-based real-time face recognition application that uses **DeepFace (ArcFace)**, **OpenCV**, and **Tkinter** to identify registered users through a webcam. The project allows users to register faces, generate facial embeddings, and perform live recognition using cosine similarity.

---

## Features

- 📷 Register new users using a webcam
- 🧠 Generate facial embeddings with ArcFace
- 💾 Store embeddings in a local database
- 🎥 Real-time face detection and recognition
- 👤 Unknown face detection using similarity threshold
- 🖥️ Simple GUI built with Tkinter

---

## Project Structure

```
.
├── dataset/                # Registered face images
├── models/
│   └── face_db.pkl         # Generated embedding database
├── register.py             # Capture images for new users
├── train_test.py           # Generate face embeddings
├── app_gui.py              # Live face recognition application
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- DeepFace
- ArcFace Model
- NumPy
- Tkinter
- Pillow

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### Install dependencies

```bash
pip install opencv-python
pip install deepface
pip install pillow
pip install numpy
```

---

## Usage

### 1. Register a User

Run:

```bash
python register.py
```

- Enter the person's name.
- Press **SPACE** to capture images.
- Around **40 images** are collected.
- Press **ESC** to finish early.

---

### 2. Train the Database

Run:

```bash
python train_test.py
```

This generates facial embeddings and stores them in:

```
models/face_db.pkl
```

---

### 3. Start Live Recognition

Run:

```bash
python app_gui.py
```

The webcam opens and recognizes registered users in real time.

---

## How It Works

1. Face images are collected for each user.
2. DeepFace extracts ArcFace embeddings from the images.
3. The average embedding is stored in a local database.
4. During live recognition:
   - Faces are detected.
   - Embeddings are generated.
   - Cosine distance is computed against the database.
   - The closest match below the threshold is displayed.
   - Otherwise, the face is labeled **Unknown**.

---

## Recognition Threshold

The recognition threshold can be adjusted in `app_gui.py`:

```python
DIST_THRESHOLD = 0.65
```

Lower values improve accuracy but may reject valid faces.

---

## Future Improvements

- Face registration through the GUI
- Multiple recognition models
- Database support (SQLite/MySQL)
- Attendance logging
- Liveness detection
- GPU acceleration
- Face mask recognition

---

## Screenshots

Add screenshots of:
- Registration window
- Live recognition
- Multiple face detection

---

## License

This project is licensed under the MIT License.

---

## Author

**Bhavin Gupta**

If you found this project useful, consider giving it a ⭐ on GitHub!
