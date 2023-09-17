import json
from deepface import DeepFace
import numpy as np
import os
import faiss

def find_key_by_value(input_dict, target_value):
    for key, values in input_dict.items():
        if target_value in values:
            return key
    return None

img_path = "/home/pi/Code/image_face/thien3.jpg"

# Đường dẫn tới tệp JSON chứa dữ liệu
input_file_path = "/home/pi/Code/gpio/embedded_vectors.json"
query_embedding = DeepFace.represent(img_path, model_name="Facenet", detector_backend="ssd")
query_embeddings = np.array(query_embedding[0]['embedding'])

image_embeddings = []

# Đọc dữ liệu từ tệp JSON
with open(input_file_path, "r") as json_file:
    embedded_vectors = json.load(json_file)

for i in list(embedded_vectors.keys()):
    for j in embedded_vectors[i]:
        embedding = np.array(j, dtype=np.float32)
        image_embeddings.append(embedding)
        
image_embeddings = np.array(image_embeddings)

# Build the FAISS index
embedding_dim = query_embeddings.shape[0]
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
index.add(image_embeddings)

# Query the index with the query embedding
k = 1 # Number of nearest neighbors to retrieve
distances, indices = index.search(np.array([query_embeddings], dtype=np.float32), k)

vector = list(image_embeddings[indices[0][0]])

name = find_key_by_value(embedded_vectors, vector)

print(name)


        
