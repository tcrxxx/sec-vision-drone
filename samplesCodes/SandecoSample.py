import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2


person = ["KNOW", "UNKNOW"]
num_class = len(person)

cap = cv2.VideoCapture(0)

detector_face = MTCNN()  # FIND FACES IN IMAGE
facenet = load_model("resources/models/facenet_keras.h5")  # FACES TO EMBEDDINGS
#model = load_model("resources/models/faces.h5")  # CLASSIFIER KNOWN OR UNKNOWN


def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)


def get_embedding(facenet, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet.predict(samples)
    return yhat[0]
