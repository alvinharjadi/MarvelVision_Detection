# Alvin Harjadi - 2301873296
# Computer Science and Mathematics

# COMP7116016 – Computer Vision
# Final Exam - Nomor 1

# Library
import numpy as np
import cv2
import os

# Path sesuai penyimpanan di device
classfier = cv2.CascadeClassifier("./Datasets/haarcascade_frontalface_default.xml")
vision_path = "./Datasets/Train/"
vision_folder_list = os.listdir(vision_path)

# List untuk menyimpan face dan class
face_list = list()
class_list = list()

# List path image keseluruhan
for idx, folder in enumerate(vision_folder_list):
    vision_full_path = vision_path + folder
    vision_image_list = os.listdir(vision_full_path)
    for img in vision_image_list:
        vision_image_path = f"{vision_full_path}/{img}"
        
        # read image, 0 berarti grayscale
        vision_image = cv2.imread(vision_image_path, 0)
        
        # detect face
        detected_faces = classfier.detectMultiScale(
            vision_image, 
            scaleFactor = 1.1, 
            minNeighbors = 5
        )

        # validasi kalau tidak ada wajah terdeteksi
        if len(detected_faces) < 1:
            continue
        # print(detected_faces)

        for face in detected_faces:
            x, y, h, w = face
            
            # crop image
            face_image = vision_image[y: y + h, x: x + w]
            
            # masukkin semua data face ke list
            face_list.append(face_image)
            class_list.append(idx)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

# Gambar dari folder pada tautan: shang chi dan secret wars
exam_path = "./Datasets/Exam/"
exam_image_list = os.listdir(exam_path)

for image in exam_image_list:
    exam_full_path = exam_path + image
    
    # bgr color
    exam_image = cv2.imread(exam_full_path)
    # gray color
    exam_image_gray = cv2.cvtColor(exam_image, cv2.COLOR_BGR2GRAY)
    
    detected_faces = classfier.detectMultiScale(
        exam_image_gray,
        scaleFactor = 1.1,
        minNeighbors = 5
    )

    # validasi kalau tidak ada wajah terdeteksi
    if len(detected_faces) < 1:
        continue

    for face in detected_faces:
        x, y, h, w = face
        # crop image
        exam_face_image = exam_image_gray[y: y + h, x: x + w]
        # recognizer memprediksi face
        res, confidence = face_recognizer.predict(exam_face_image)
        if (confidence > 100):
            exam_image = cv2.rectangle(
                exam_image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            # membuat text untuk ditampilkan
            text = f"{vision_folder_list[res]} : {confidence}"
            exam_image = cv2.putText(
                exam_image, 
                text, 
                (x, y - 10), 
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.imshow("Alvin Harjadi - 2301873296", exam_image)
            cv2.waitKey(0)