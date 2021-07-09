from faces import utilsFaces
from faces.utilsFaces import *
from tello.telloUtils import *
import cv2
import time
from mtcnn.mtcnn import MTCNN

#########################################################
# DEFINITIONS
#########################################################

# System
NAME_SYS = "SECURITY VISION DRONE"

# Models
detectorMTCNN = MTCNN()  # FIND FACES IN IMAGE

# Tello
w_img, h_img = 360, 240
pid = [0.4, 0.4, 0]
pError = 0
startCounter = 1  # for no Flight 1   - for flight 0

# Bounding Box
init_classe = 0
classe_label = ["UNKNOWN", "KNOW"]
num_class = len(classe_label)
color = [(0, 0, 255), (255, 0, 0)] #BGR
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
pos_user_label = 10
thickness = 1

print("Initialize Drone")
myDrone = initializeTello()


def drawBoundingBox(pimg, px1, py1, px2, py2):
    cv2.rectangle(pimg, (px1, py1), (px2, py2), color[init_classe], 2)
    cv2.putText(pimg,
                user,
                (px1, py1 - pos_user_label),
                font,
                fontScale=font_scale,
                color=color[init_classe],
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

    # DETECT FACES - MTCNN
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

            user = str(classe_label[init_classe])  # TODO: Change to classe returned by model
            print("User finded:" + user)

            # DRAW BOUNDING BOX
            drawBoundingBox(img, x1, y1, x2, y2)

    cv2.imshow(NAME_SYS, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        if startCounter == 0:
            myDrone.land()
        break

