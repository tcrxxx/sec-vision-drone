from faces import Faces
from faces.Faces import *
from tello.VideoCaptureTool import *
import cv2

# Tello
w, h = 360, 240
pid = [0.4, 0.4, 0]
pError = 0
startCounter = 1  # for no Flight 1   - for flight 0
myDrone = initializeTello()

while True:

    ## Flight
    if startCounter == 0:
        myDrone.takeoff()
        startCounter = 1

    ## Step 1
    #img = telloGetFrame(myDrone, w, h)
    #img = telloGetFrame(myDrone, w, h)
    ## Step 2
    # img, info = findFace(img)
    execute(myDrone)
    ## Step 3
    #pError = trackFace(myDrone, info, w, pid, pError)
    # print(info[0][0])
    #cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        myDrone.land()
        break
