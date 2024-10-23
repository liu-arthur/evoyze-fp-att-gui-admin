import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# Initialize the ArcFace-based model for face analysis
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Create a directory for saving embeddings if it doesn't exist
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

def capture_faces(poses, images_per_pose=20):
    cap = cv2.VideoCapture(0)
    all_images = []

    for pose in poses:
        print(f"\nPlease {pose}. Capturing will start in 3 seconds...")
        cv2.waitKey(3000)  # Wait for 3 seconds before starting capture
        images_captured = 0

        while images_captured < images_per_pose:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                continue

            faces = app.get(frame)
            if faces:
                # Select the largest face
                faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                bbox = faces[0].bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                images_captured += 1
                all_images.append(frame)
                print(f"Captured image {images_captured}/{images_per_pose} for '{pose}' pose.")

            cv2.imshow(f"Pose: {pose}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Exit on 'esc'
                cap.release()
                cv2.destroyAllWindows()
                return all_images

        print(f"Finished capturing images for '{pose}' pose.")

    cap.release()
    cv2.destroyAllWindows()
    return all_images

# Function to normalize the embeddings
def normalize(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Function to save the embeddings
def save_embeddings(images, name):
    embeddings = []
    for img in images:
        faces = app.get(img)
        if faces:
            embedding = normalize(faces[0].embedding)  # Normalize embedding
            print(f"Generated embedding for {name}, shape: {embedding.shape}")  # Debugging statement
            embeddings.append(embedding)

    if embeddings:
        # Average the embeddings to get a representative embedding
        average_embedding = np.mean(embeddings, axis=0)
        # Save the average embedding as a .npy file
        np.save(f'embeddings/{name}_embedding.npy', average_embedding)
        print(f"Saved average embedding for {name}.")
    else:
        print(f"No valid embeddings generated for {name}.")

if __name__ == "__main__":
    name = input("Enter your name: ")
    print("We will capture images in different poses to improve the training.")

    # Define the poses and number of images per pose
    poses = [
        "look straight at the camera",
        "turn your face to the left",
        "turn your face to the right",
        "look up",
        "look down"
    ]

    total_images = 0
    images_per_pose = 20  # Adjust as needed
    total_images = images_per_pose * len(poses)
    print(f"\nTotal images to capture: {total_images}")

    images = capture_faces(poses, images_per_pose=images_per_pose)

    if images:
        save_embeddings(images, name)
    else:
        print("No images were captured.")
