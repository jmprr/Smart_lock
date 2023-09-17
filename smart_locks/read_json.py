import json
from deepface import DeepFace
import numpy as np
import os

# Đường dẫn tới tệp JSON chứa dữ liệu
input_file_path = "/home/pi/Code/gpio/embedded_vectors.json"

# Đọc dữ liệu từ tệp JSON
with open(input_file_path, "r") as json_file:
    embedded_vectors = json.load(json_file)

img_path = "/home/pi/Code/deepface/Deepface/Face-detection-SSD/database/thien2.jpg"

query_embedding = DeepFace.represent(img_path, model_name="Facenet512", detector_backend="ssd")
query_embeddings = query_embedding[0]['embedding']

print("Enter Member: ")
name = str(input())

if name in embedded_vectors:
    embedded_vectors['{}'.format(name)].append(query_embeddings)
else:
    new_data = {
        '{}'.format(name) : [query_embeddings]
    }

    embedded_vectors.update(new_data)

with open(input_file_path, "w") as json_file:
    json.dump(embedded_vectors, json_file, indent=4)
