import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Initialize the ArcFace-based model for face analysis
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Create a directory for saving embeddings if it doesn't exist
if not os.path.exists('embeddings'):
    os.makedirs('embeddings')

def capture_faces(num_images=20):
    cap = cv2.VideoCapture(0)
    images = []
    while len(images) < num_images:
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
            cv2.imshow("Press 'q' to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and faces:
            images.append(frame)
            print(f"Captured image {len(images)}/{num_images}")
        if key == 27:  # Exit on 'esc'
            break

    cap.release()
    cv2.destroyAllWindows()

    # Check if the number of images is 20
    if len(images) != 20:
        raise KeyError("You must capture exactly 20 images.")
    
    return images

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
            embeddings.append(embedding)

    if embeddings:
        # Convert embeddings list to a NumPy array
        embeddings_array = np.vstack(embeddings)
        # Average the embeddings to get a representative embedding
        average_embedding = np.mean(embeddings_array, axis=0)
        # Save the average embedding as a .npy file
        np.save(f'embeddings/{name}_embedding.npy', average_embedding)
        print(f"Saved average embedding for {name}.")
        return embeddings_array  # Return the embeddings array
    else:
        print(f"No valid embeddings generated for {name}.")
        return None

from sklearn.decomposition import PCA

def reduce_embeddings(embeddings):
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(reduced_embeddings, name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the embeddings
    ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c='b', marker='o', label=name
    )

    # Set labels
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')

    # Set title
    ax.set_title(f'3D PCA of {name}\'s Face Embeddings')

    # Show legend
    ax.legend()

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Display additional information before asking for input
    print("Welcome to the Attendance Face Capture System.")
    print("Please ensure you are in a well-lit area.")
    print("You will be prompted to enter either your name or employee ID.")
    print("NOTE: The name or ID provide MUST/ going to registered in the attendance module.")
    print("Press 'q' to capture photo and 'Esc' to exit.")
    name = input("Please enter employee name or ID: ")
    images = capture_faces(num_images=20)

    if images:
        embeddings = save_embeddings(images, name)
        if embeddings is not None:
            reduced_embeddings = reduce_embeddings(embeddings)
            plot_embeddings(reduced_embeddings, name)
