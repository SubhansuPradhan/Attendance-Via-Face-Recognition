# recognize.py (DeepFace + OpenCV detector)
import cv2
import pickle
import argparse
import os
from datetime import datetime
import csv
import numpy as np
from deepface import DeepFace

def mark_attendance(csv_path, name):
    now = datetime.now().isoformat(sep=' ', timespec='seconds')
    header = ["name", "time"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([name, now])
    print(f"[ATTENDANCE] Marked {name} at {now}")

def main(encodings_path, tolerance, scale, output_csv, model_name, detector_backend):
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
    known_embeddings = data["embeddings"]
    known_names = data["names"]

    seen_names = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Can't open webcam.")
        return

    print(f"[INFO] Starting recognition with {model_name} and {detector_backend}. Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        try:
            detections = DeepFace.extract_faces(
                img_path=small,
                detector_backend=detector_backend,
                enforce_detection=False
            )
        except Exception as e:
            print(f"[WARN] Detection error: {e}")
            detections = []

        for det in detections:
            facial_area = det["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

            face_img = small[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            # Get embedding for this face
            emb = DeepFace.represent(
                face_img,
                model_name=model_name,
                enforce_detection=False
            )[0]["embedding"]

            # Match with known embeddings
            name = "Unknown"
            if len(known_embeddings) > 0:
                distances = np.linalg.norm(np.array(known_embeddings) - emb, axis=1)
                best_idx = np.argmin(distances)
                if distances[best_idx] <= tolerance:
                    name = known_names[best_idx]

            # Scale coordinates back to original frame
            factor = 1.0 / scale
            x, y, w, h = int(x * factor), int(y * factor), int(w * factor), int(h * factor)

            # Draw rectangle and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mark attendance once
            if name != "Unknown" and name not in seen_names:
                mark_attendance(output_csv, name)
                seen_names.add(name)

        cv2.imshow("Recognition - Press q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Recognition stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encodings", default="encodings.pkl", help="Path to encodings file.")
    parser.add_argument("--tolerance", type=float, default=25, help="Distance threshold for face matching.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for faster detection (0.25-1.0).")
    parser.add_argument("--output", default="attendance.csv", help="Attendance CSV output path.")
    parser.add_argument("--model-name", default="VGG-Face", help="DeepFace model to use (VGG-Face, Facenet, ArcFace, etc.).")
    parser.add_argument("--detector-backend", default="opencv", help="Face detector backend (opencv, mtcnn, dlib, retinaface, mediapipe, ssd).")
    args = parser.parse_args()
    main(args.encodings, args.tolerance, args.scale, args.output, args.model_name, args.detector_backend)
