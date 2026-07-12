import cv2
import os

name = input("Enter person's name: ").strip()
path = os.path.join("dataset", name)
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0
required = 40

print("[INFO] Press SPACE to capture")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Register", frame)
    key = cv2.waitKey(1)

    if key == 32:
        cv2.imwrite(os.path.join(path, f"{count}.jpg"), frame)
        count += 1
        print("Captured:", count)

    if key == 27 or count >= required:
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE]")