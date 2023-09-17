from gpiozero import Button
from gpiozero import LED
from deepface import DeepFace
import numpy as np
import cv2
import os
import faiss
import json
import time
from function_for_pi import recognition, add_new_member
import subprocess

led = LED(4)

button = Button(17)

command = "scp -r ./file.txt duong@172.20.10.3:/home/duong/Documents/Smart_lock/storage"

while True:
    print("Press")
    button.wait_for_press()
    led.on()
    
    subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    time.sleep(1.25)
    img_path = '/home/pi/Code/image_face/captured_image.jpg'
    
    print('loading...')
    add_new_member(image_path = img_path)
    
    led.off()