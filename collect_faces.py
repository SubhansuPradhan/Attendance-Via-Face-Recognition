# collect_faces.py
import os
import cv2
import time
import argparse
from deepface import DeepFace

def clamp(v, low, high):
    return max(low, min(high, v))

def main(name, count, save_dir, scale):
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    saved = 0
    print(f"[INFO] Starting capture for '{name}'. Press 'q' to quit early.")
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Can't read frame from webcam.")
            break

        # resize for faster processing
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        try:
            detections = DeepFace.extract_faces(
                img_path=small, detector_backend="opencv", enforce_detection=False
            )
        except Exception as e:
            print(f"[WARN] Detection error: {e}")
            detections = []

        for det in detections:
            fa = det["facial_area"]
            x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

            factor = 1.0 / scale
            x = int(x * factor)
            y = int(y * factor)
            w = int(w * factor)
            h = int(h * factor)

            top_o, left_o, bottom_o, right_o = y, x, y + h, x + w
            top_o = clamp(top_o, 0, frame.shape[0])
            bottom_o = clamp(bottom_o, 0, frame.shape[0])
            left_o = clamp(left_o, 0, frame.shape[1])
            right_o = clamp(right_o, 0, frame.shape[1])

            face_img = frame[top_o:bottom_o, left_o:right_o]
            if face_img.size == 0:
                continue

            fname = os.path.join(person_dir, f"{name}_{saved+1:03d}.jpg")
            cv2.imwrite(fname, face_img)
            saved += 1
            print(f"[SAVED] {fname}")
            time.sleep(0.25)

            cv2.rectangle(frame, (left_o, top_o), (right_o, bottom_o), (0,255,0), 2)
            cv2.putText(frame, f"{saved}/{count}", (left_o, top_o-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            if saved >= count:
                break

        cv2.imshow("Collect Faces - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Collected {saved} images for '{name}' in '{person_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Person name (folder name in dataset).")
    parser.add_argument("--count", type=int, default=50, help="Number of face images to collect.")
    parser.add_argument("--save-dir", default="dataset", help="Dataset root folder.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for faster detection (0.25-1.0).")
    args = parser.parse_args()
    main(args.name, args.count, args.save_dir, args.scale)
