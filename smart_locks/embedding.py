import numpy as np
from deepface import DeepFace
import os

# Load an image for which you want to generate embeddings
folder_path = "./database"

# Generate embeddings for the image
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

image_embeddings = []

for image_file in image_files:
    img_path = os.path.join(folder_path, image_file)
    embedding = DeepFace.represent(img_path, model_name="Facenet", detector_backend="ssd")
    embeddings = np.array(embedding[0]['embedding'], dtype=np.float32)
    image_embeddings.append(embeddings)
    
    print("Image:", image_file)
    #print("Face embedding:", embedding[0]['embedding'])
    print("Face embedding", embeddings)
    print("-" * 30)
    
print(image_embeddings)
