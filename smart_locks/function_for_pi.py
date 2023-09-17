from deepface import DeepFace
import numpy as np
import os
import faiss
import json
import time
import cv2
import RPi.GPIO as GPIO
import time

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)

def open_lock():
    GPIO.output(18, 1)
    time.sleep(2)
    GPIO.output(18, 0)

def find_key_by_value(input_dict, target_value):
    for key, values in input_dict.items():
        if target_value in values:
            return key
    return None

def face_detection(image_path):
    threshold = 0.5 # human face's confidence threshold

    prototxt_file = os.path.join('./Face_detection/SSD_deploy.prototxt')
    caffemodel_file = os.path.join('./Face_detection/model.caffemodel')
    net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)

    image = cv2.imread(image_path)
    origin_h, origin_w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    tic = time.time()
    net.setInput(blob)
    detections = net.forward()
    print('net forward time: {:.4f}'.format(time.time() - tic))
    # detection.shape = (1,1,num_bbox,7) with 7 is 2 output is face or non_face and (x,y,w,h,conf) 

    bounding_boxs = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            bounding_boxs.append(list(bounding_box.astype('int')))

            # x_start, y_start, x_end, y_end = largest_face.astype('int')
            # cropped_image = image[y_start:y_end, x_start:x_end]
            # cv2.imwrite('./test_image/face{}.jpg'.format(i), cropped_image)

    print(bounding_boxs)

    largest_face = None
    largest_area = 0
    
    for i in bounding_boxs:
        for (x, y, x1, y1) in [i]:
            area = (x1-x) * (y1-y)
            if area > largest_area:
                largest_area = area
                largest_face = [x, y, x1, y1]
                
    if not bounding_boxs:
        return None
    else:
        bounding_boxs.remove(largest_face)

        for j in bounding_boxs:
            for (x, y, x1, y1) in [j]:
                cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 0), -1)
                
        return image_path, cv2.imwrite(image_path, image)
    


def add_new_member(img_path, database = "/home/pi/Code/gpio/embedded_vectors.json"):
    with open(database, "r") as json_file:
        embedded_vectors = json.load(json_file)
        
    imgs_path = face_detection(image_path=img_path)
    
    if imgs_path == None:
        os.remove(img_path)
        print("Please try Again")
    
    else:
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

        with open(database, "w") as json_file:
            json.dump(embedded_vectors, json_file, indent=4)
        
        os.remove(img_path)
        
        return print("Successfully added")
        
        
def embed(image_path):
    query_embedding = DeepFace.represent(image_path, model_name="Facenet512", detector_backend="ssd")
    query_embeddings = np.array(query_embedding[0]['embedding'])
    
    return query_embeddings


def recognition(image_path, database = "/home/pi/Code/gpio/embedded_vectors.json"):
    query_embeddings = embed(image_path)

    image_embeddings = []

    with open(database, "r") as json_file:
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

    if distances[0][0] > 300:
        print("Get out")
    else:
        vector = list(image_embeddings[indices[0][0]])

        name = find_key_by_value(embedded_vectors, vector)

        return print(name), open_lock() 

