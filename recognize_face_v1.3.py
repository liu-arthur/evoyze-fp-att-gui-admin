import cv2
import numpy as np
import os
import threading
from queue import Queue
from insightface.app import FaceAnalysis
import logging
import time

# Configure logging (set to WARNING to reduce overhead)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize InsightFace for ArcFace-based recognition
# Use only CPU provider for less computing power consumption
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Load stored embeddings
def load_embeddings(embeddings_dir='embeddings'):
    embeddings = {}
    if not os.path.exists(embeddings_dir):
        logging.warning(f"Embeddings directory '{embeddings_dir}' does not exist.")
        return embeddings

    for file in os.listdir(embeddings_dir):
        if file.endswith('_embedding.npy'):
            name = file.split('_embedding.npy')[0]
            embeddings_path = os.path.join(embeddings_dir, file)
            embeddings[name] = np.load(embeddings_path)
            logging.info(f"Loaded embedding for {name}")
    return embeddings

# Recognize face using ArcFace embedding comparison
def recognize_face(live_embedding, stored_embeddings, threshold=0.6):
    recognized_name = "Unknown"
    max_similarity = -1

    # Normalize the live embedding
    live_embedding_norm = live_embedding / np.linalg.norm(live_embedding)

    for name, stored_embedding in stored_embeddings.items():
        # Normalize the stored embedding
        stored_embedding_norm = stored_embedding / np.linalg.norm(stored_embedding)
        # Compute cosine similarity
        similarity = np.dot(live_embedding_norm, stored_embedding_norm)
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = name

    if max_similarity < threshold:
        recognized_name = "Unknown"

    return recognized_name, max_similarity

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open webcam.")
        return

    # Reduce frame size to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # You can adjust this value
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # You can adjust this value

    # Load stored embeddings
    stored_embeddings = load_embeddings()

    if not stored_embeddings:
        logging.error("No embeddings found. Exiting.")
        return

    # Queue for frames to be processed
    frame_queue = Queue(maxsize=5)
    # Dictionary to hold recognition results
    recognition_results = {}

    # Flag to stop threads
    stop_event = threading.Event()

    def face_recognition_thread():
        while not stop_event.is_set():
            if not frame_queue.empty():
                frame = frame_queue.get()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = app.get(rgb_frame)

                results = []
                for face in faces:
                    live_embedding = face.embedding
                    recognized_name, similarity = recognize_face(live_embedding, stored_embeddings)

                    # Store results
                    results.append({
                        'bbox': face.bbox.astype(int),
                        'name': recognized_name,
                        'similarity': similarity
                    })
                recognition_results['results'] = results
            else:
                time.sleep(0.01)  # Slight delay to prevent CPU overuse

    # Start the face recognition thread
    threading.Thread(target=face_recognition_thread, daemon=True).start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to capture image")
                break

            # Put the frame in the queue if it's not full
            if not frame_queue.full():
                frame_queue.put(frame.copy())

            # Display the frame and overlay recognition results if available
            if 'results' in recognition_results:
                for result in recognition_results['results']:
                    x1, y1, x2, y2 = result['bbox']
                    recognized_name = result['name']
                    similarity = result['similarity']

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Prepare label with name and similarity score
                    if recognized_name != "Unknown":
                        label = f"{recognized_name} ({similarity:.2f})"
                    else:
                        label = recognized_name

                    # Put the recognized name above the bounding box
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the video feed with recognition
            cv2.imshow("Face Recognition", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Release resources
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources released. Exiting application.")

if __name__ == "__main__":
    main()
