from deepface import DeepFace
import numpy as np
import os
import faiss

# Load an image for which you want to generate embeddings
folder_path = "./database"
img_path = "/home/pi/Code/image_face/face.jpg"

query_embedding = DeepFace.represent(img_path, model_name="Facenet512", detector_backend="ssd")
query_embeddings = np.array(query_embedding[0]['embedding'])

# Generate embeddings for the image
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

image_embeddings = []

for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    embedding = DeepFace.represent(img_path, model_name="Facenet512", detector_backend="ssd")
    embeddings = np.array(embedding[0]['embedding'], dtype=np.float32)
    image_embeddings.append(embeddings)
    
    print("Image:", image_file)
    print("Face embedding:", embedding[0]['embedding'])
    print("Face embedding shape", embeddings.shape)
    print("-" * 30)
    
image_embeddings = np.array(image_embeddings)

print("Query embedding shape:", query_embeddings.shape)  # It should be (embedding_dim,)

# Check the shape of image embeddings
print("Image embeddings shape:", image_embeddings.shape)  # It should be (num_embeddings, embedding_dim)

# Build the FAISS index
embedding_dim = query_embeddings.shape[0]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
index.add(image_embeddings)

# Query the index with the query embedding
k = 9 # Number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_embeddings], dtype=np.float32), k)

print("Nearest neighbors:")
for i in range(k):
    nearest_image_path = os.path.join(folder_path, os.listdir(folder_path)[indices[0][i]])
    print(f"{i + 1}: {nearest_image_path} (Distance: {distances[0][i]})")
