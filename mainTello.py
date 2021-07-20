import numpy as np

from utils.utilsFaces import extract_face, get_embedding
from utils.telloUtils import *
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

#########################################################
# DEFINITIONS
#########################################################

# System
NAME_SYS = "SECURITY VISION DRONE"

# Models
detectorMTCNN = MTCNN()  # FIND FACES IN IMAGE
facenet = load_model("resources/models/facenet_keras.h5") # FACES TO EMBEDDINGS
model = load_model("resources/models/faces_percep.h5")  # CLASSIFIER KNOWN OR UNKNOWN

# Tello
w_img, h_img = 360, 240
pid = [0.4, 0.4, 0]
pError = 0
startCounter = 1  # for no Flight 1   - for flight 0

# Bounding Box
init_classe = 0
class_label = ["KNOW", "UNKNOWN"]
num_class = len(class_label)
color = [(0, 255, 0), (0, 0, 255)] #BGR
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
pos_user_label = 10
thickness = 1

print("Initialize Drone")
myDrone = initializeTello()


def drawBoundingBox(pimg, user, predicted_class, px1, py1, px2, py2):
    cv2.rectangle(pimg, (px1, py1), (px2, py2), color[predicted_class], 2)
    cv2.putText(pimg,
                user,
                (px1, py1 - pos_user_label),
                font,
                fontScale=font_scale,
                color=color[predicted_class],
                thickness=thickness)


while True:
    ## Flight
    print("Check if fly need...")
    if startCounter == 0:
        print("Takeoff...")
        myDrone.takeoff()
        startCounter = 1

    print("Read stream drone...")
    img = myDrone.get_frame_read().frame
    img = cv2.resize(img, (w_img, h_img))

    # MODEL: DETECT FACES - MTCNN
    print("Find faces (MTCNN)...")
    faces = detectorMTCNN.detect_faces(img)
    print("Find " + str(len(faces)) + " faces...")

    for face in faces:
        print("Face identified:" + str(face))
        print("Face confidence:" + str(face['confidence'] * 100))

        # Capture frame only confidence > 98
        if (face['confidence'] * 100) >= 98:
            x1, y1, w, h = face['box']
            x2, y2 = x1 + w, y1 + h
            print("x1:" + str(x1))
            print("x2:" + str(x2))
            print("y1:" + str(y1))
            print("y2:" + str(y2))
            print("w:" + str(w))
            print("h:" + str(h))

            # PRE PROCESSING ?

            # MODEL: GET EMBEDDINGS
            face_extracted = extract_face(img, face['box'])
            face_extracted = face_extracted.astype("float32")/255
            embeddings = get_embedding(facenet, face_extracted)
            print("Embedding: {}".format(str(embeddings)))

            # MODEL: RECOGNIZE FAC
            tensor = np.expand_dims(embeddings, axis=0)

            # Normalize
            from sklearn.preprocessing import Normalizer
            norm = Normalizer(norm="l2")
            sample = norm.transform(tensor)

            predicted_class = model.predict_classes(tensor)[0]

            prob = model.predict_proba(tensor)
            prob = prob[0][predicted_class]*100

            # Recognize only prob return greater than 98
            if prob >= 98:
                # DEFINE USER BY FACE RECOGNIZE
                user = str(class_label[predicted_class])
                print("User finded:" + user)

                # DRAW BOUNDING BOX
                drawBoundingBox(img, user, predicted_class, x1, y1, x2, y2)

    cv2.imshow(NAME_SYS, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        if startCounter == 0:
            myDrone.land()
        break

