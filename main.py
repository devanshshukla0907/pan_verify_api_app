from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form
import cv2
import numpy as np
import pytesseract
from io import BytesIO


app = FastAPI()


def extract_pan_number(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 6')
    return text

def detect_face(image: np.ndarray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return gray[y:y+h, x:x+w]

def compare_images(image1: np.ndarray, image2: np.ndarray) -> float:
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

@app.post("/verify_pan")
async def verify_pan(
    pan_number: str = Form(...),
    pan_card: UploadFile = File(...),
    person_image: UploadFile = File(...),
):
    pan_card_image = np.asarray(bytearray(await pan_card.read()), dtype=np.uint8)
    person_image_data = np.asarray(bytearray(await person_image.read()), dtype=np.uint8)

    pan_card_img = cv2.imdecode(pan_card_image, cv2.IMREAD_COLOR)
    person_img = cv2.imdecode(person_image_data, cv2.IMREAD_COLOR)
    
    extracted_pan = extract_pan_number(pan_card_img)
    if pan_number not in extracted_pan:
        return {"verified": False, "reason": "PAN number mismatch"}
    
    face_pan = detect_face(pan_card_img)
    face_person = detect_face(person_img)
    
    if face_pan is None or face_person is None:
        return {"verified": False, "reason": "Face not detected"}
    
    similarity = compare_images(face_pan, face_person)
    
    if similarity > 0.5:
        return {"verified": True, "message": "PAN number and face matched"}
    else:
        return {"verified": False, "reason": "Face mismatch"}

@app.get("/")
async def root():
    return {"message": "Hello World"}
