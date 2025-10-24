# train_encodings.py
import os
import pickle
from deepface import DeepFace


def train(dataset_path="dataset", encodings_path="encodings.pkl", model_name="VGG-Face"):
    known_embeddings = []
    known_names = []

    print(f"[INFO] Training model using {model_name} embeddings...")

    # loop over dataset
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"[INFO] Processing {person_name}...")

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            try:
                # Represent face as an embedding vector
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=model_name,
                    enforce_detection=False
                )

                if isinstance(embedding, list) and len(embedding) > 0:
                    vec = embedding[0]["embedding"]
                    known_embeddings.append(vec)
                    known_names.append(person_name)

            except Exception as e:
                print(f"[WARN] Could not process {img_path}: {e}")

    # save encodings + names
    data = {"embeddings": known_embeddings, "names": known_names}
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[DONE] Training complete. Encodings saved to {encodings_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder.")
    parser.add_argument("--encodings", default="encodings.pkl", help="Output encodings file.")
    parser.add_argument("--model-name", default="VGG-Face", help="DeepFace model (VGG-Face, Facenet, ArcFace, etc.)")
    args = parser.parse_args()

    train(args.dataset, args.encodings, args.model_name)
