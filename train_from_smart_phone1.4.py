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

def load_images_from_folder(folder):
    images = []
    image_names = []
    supported_formats = ('.jpg', '.jpeg', '.png')
    for filename in os.listdir(folder):
        if filename.lower().endswith(supported_formats):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                image_names.append(filename)
            else:
                print(f"Failed to load image {filename}")
    return images, image_names

# Function to normalize the embeddings
def normalize(embedding):
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm != 0 else embedding

# Function to save the embeddings
def save_embeddings(images, image_names, name):
    embeddings = []
    valid_image_names = []
    for img, img_name in zip(images, image_names):
        faces = app.get(img)
        if faces:
            embedding = normalize(faces[0].embedding)  # Normalize embedding
            embeddings.append(embedding)
            valid_image_names.append(img_name)
        else:
            print(f"No face detected in image {img_name}")

    if embeddings:
        # Convert embeddings list to a NumPy array
        embeddings_array = np.vstack(embeddings)
        # Average the embeddings to get a representative embedding
        average_embedding = np.mean(embeddings_array, axis=0)
        # Save the average embedding as a .npy file
        np.save(f'embeddings/{name}_embedding.npy', average_embedding)
        print(f"Saved average embedding for {name}.")
        return embeddings_array, valid_image_names  # Return the embeddings array and valid image names
    else:
        print(f"No valid embeddings generated for {name}.")
        return None, None

def reduce_embeddings(embeddings):
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(reduced_embeddings, name, image_names=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the embeddings
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2],
        c='b', marker='o', label=name
    )

    # Optionally annotate points with image names
    if image_names:
        for i, img_name in enumerate(image_names):
            ax.text(
                reduced_embeddings[i, 0],
                reduced_embeddings[i, 1],
                reduced_embeddings[i, 2],
                img_name,
                fontsize=8
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
    print("Welcome to the Attendance Face Capture System.")
    print("You will be prompted to enter either your name or employee ID.")
    print("NOTE: The name or ID provide MUST/ going to registered in the attendance module.")

    images_folder = 'phone'

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    images, image_names = load_images_from_folder(images_folder)

    if images:
        name = input("Please enter employee name or ID: ")

        embeddings, valid_image_names = save_embeddings(images, image_names, name)
        if embeddings is not None:
            reduced_embeddings = reduce_embeddings(embeddings)
            plot_embeddings(reduced_embeddings, name, valid_image_names)
    else:
        print("No images found in the folder or failed to load images.")
