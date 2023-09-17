from gpiozero import Button
from gpiozero import LED
from deepface import DeepFace
import numpy as np
import cv2
import os
import faiss
import json
import time
from function_for_pi import recognition, face_detection, add_new_member
import subprocess
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

button_pin_1 = 17
button_pin_2 = 12
led_pin_1 =4
led_pin_2 = 1

GPIO.setup(button_pin_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(button_pin_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(led_pin_1, GPIO.OUT)
GPIO.setup(led_pin_2, GPIO.OUT)

command = "scp -r ./file.txt duong@172.20.10.3:/home/duong/Documents/Smart_lock/storage"

print("Press")

while True:
    button_state_1 = GPIO.input(button_pin_1)
    button_state_2 = GPIO.input(button_pin_2)
    
    if button_state_1 == GPIO.LOW:
        GPIO.output(led_pin_1, GPIO.HIGH)
        
        subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        time.sleep(1.5)
        img_path = './image_face/captured_image.jpg'
        
        print('loading...')
        imgs_path = face_detection(image_path=img_path)
        
        if imgs_path == None:
            #os.remove(img_path)
            print('Please try again')
        else:
            recognition(image_path = img_path)
            #os.remove(img_path)
            print('Finish')
        
    else:
        GPIO.output(led_pin_1, GPIO.LOW)
    
    if button_state_2 == GPIO.LOW:
        GPIO.output(led_pin_2, GPIO.HIGH)
    
        subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        time.sleep(1.5)
        img_path = './captured_image.jpg'
        
        print('loading...')
        add_new_member(img_path = img_path)
        
    else:
        GPIO.output(led_pin_2, GPIO.LOW)
    
    
    
    


