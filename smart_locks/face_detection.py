import numpy as np
import os
import time
import cv2
import time

image_path = "./image_face/face.jpg"

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

