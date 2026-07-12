# Dataset

This folder stores the images used to train the face recognition model.

## Structure

```
dataset/
├── Person1/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── Person2/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
```

- Create one folder per person.
- Name the folder using the person's name.
- Capture around **40 images** for each person using `register.py`.
- Images should clearly show the person's face under different angles and lighting conditions.

> **Note:** This folder is intentionally left empty in the repository. Add your own images before running `train_test.py` to generate the face embedding database.
