from PIL import Image
from os import listdir
from os.path import isdir
from numpy import asarray, expand_dims
import numpy as np
from keras_facenet import FaceNet
import pandas as pd


facenet = FaceNet()


def load_face(filename):
    image = Image.open(filename)
    image = image.convert("RGB")
    return asarray(image)


def load_dir_faces(src_dir):
    faces = list()
    for filename in listdir(src_dir):
        try:
            path = src_dir + filename
            faces.append(load_face(path))
        except Exception as e:
            print("The error {} was ocurred to process img:".format(e, path))

    return faces


def load_photos(src_dir):
    x, y = list(), list()

    for subdir in listdir(src_dir):
        path = src_dir + subdir + "/"

        if not isdir(path):
            continue

        faces = load_dir_faces(path)

        labels = [subdir for _ in range(len(faces))]

        print("Load {} faces of class: {}".format(len(faces), subdir))

        x.extend(faces)
        y.extend(labels)

    return asarray(x), asarray(y)


def get_embedding(facenet, face_pixels):
    #Padronizacao das faces
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    #yhat = facenet.predict(samples)
    yhat = facenet.embeddings(samples)
    return yhat[0]


####################################################
# MAIN
####################################################


trainX, trainY = load_photos("./faces/")

print("trainX.shape:" + str(trainX.shape))
print("trainY.shape:" + str(trainY.shape))

newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(facenet, face_pixels)
    newTrainX.append(embedding)

newTrainX = asarray(newTrainX)
print("newTrainX.shape:" + str(newTrainX.shape))

df = pd.DataFrame(data=newTrainX)

df['target'] = trainY

print(df)

df.to_csv('../resources/models/faces.csv')