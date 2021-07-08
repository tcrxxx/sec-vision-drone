import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tello.VideoCaptureTool import *

NAME_SYS = "SECURITY VISION DRONE"

detector = MTCNN()  # FIND FACES IN IMAGE

init_classe = 0

# Bounding Box definitions
classe_label = ["UNKNOWN", "KNOW"]
num_class = len(classe_label)
color = [(0, 0, 255), (255, 0, 0)] #BGR
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
pos_user_label = 10
thickness = 1

def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)


def get_embedding(facenet,face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean = face_pixels.mean()
    std = face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels,axis=0) # Expand dimens [] => [[]]
    yhat = facenet.predict(samples)
    return yhat[0]  # Face represented


def execute(myDrone):
    while True:
            #_, frame = img_frame.read()
            frame = telloGetFrame(myDrone)

            # DETECT FACES - MTCNN
            faces = detector.detect_faces(frame)

            for face in faces:
                # Utiliza apenas faces com confiança > 98
                if (face['confidence']*100) >= 98:
                    x1, y1, w, h = face['box']
                    x2, y2 = x1+w, y1+h
                    face = extract_face(frame, face['box'])
                    face = face.astype('float32')/255

                    # EMBEDDINGS EXTRACT
                    # emb = get_embedding(facenet,face)
                    # tensor = np.expand_dims(emb,axis=0) # Expand dimens [] => [[]]

                    # PREDICT
                    #classe = model.predict_classes(tensor)[0]
                    #prob = model.predict_proba(tensor)
                    #prob = prob[0][classe]*100

                    user = str(classe_label[init_classe]) # TODO: Change to classe returned by model

                    # DRAW BOUNDING BOX
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color[init_classe], 2)
                    cv2.putText(frame,
                                user,
                                (x1, y1-pos_user_label),
                                font,
                                fontScale=font_scale,
                                color=color,
                                thickness=thickness)

            cv2.imshow(NAME_SYS, frame)

            key = cv2.waitKey(1)
            if key ==27: #ESC
                break

    cv2.destroyAllWindows()
