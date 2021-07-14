from mtcnn import MTCNN
from PIL import Image
from os import listdir, walk
from os.path import isdir
from numpy import asarray

# Faces identify
MIN_CONFIDENCE = 90
detector = MTCNN()

# Use: linux,mac,windows="\\"
FILEPATH_DIVISOR = '/'

# Sample final dir:  ./photos/myName
# Sample final dir:  ./faces/myName
FACES_DIR = "./faces/"
PHOTOS_DIR = "./fotos/"


def summary_files(path):
    print("############################################")
    totalFiles = 0
    totalDir = 0

    for base, dirs, files in walk(path):
        print('Searching in : ', base)
        for directories in dirs:
            totalDir += 1
        for Files in files:
            totalFiles += 1

    print('Total number of files', totalFiles)
    print('Total Number of directories', totalDir)
    # print('Total:', (totalDir + totalFiles))


def extract_faces(filename, size=(160, 160)):
    print("############################################")
    print("Try extract face into filename:" + filename)

    img_raw = Image.open(filename)
    img_raw = img_raw.convert('RGB')

    array_img = asarray(img_raw)
    result = detector.detect_faces(array_img)

    print("Detected Faces:" + str(len(result)))
    if len(result) == 0:
        return

    print("Confidence:" + str(result[0]['confidence'] * 100))
    # Ignore faces with not confidence
    if (result[0]['confidence'] * 100) < MIN_CONFIDENCE:
        print("Ignored")
        return

    print("Lets extract")

    # Return first face only
    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height

    # Extract face box
    face = array_img[y1:y2, x1:x2]

    image_face_box = Image.fromarray(face)
    image_face_box = image_face_box.resize(size)

    return image_face_box


def load_photos(root_dir, target_dir):
    print("Photos dir:" + root_dir)
    print("Faces dir:" + target_dir)

    for filename in listdir(root_dir):
        path = root_dir + filename
        path_target = target_dir + filename
        path_target_fliped = target_dir + "fliped-" + filename

        try:
            face = extract_faces(path)
            if face is not None:
                face.save(path_target, 'JPEG', quality=100, optimize=True, progressive=True)
                flip_and_save(face, path_target_fliped)
        except Exception as e:
            print("The error {} was ocurred to process img:".format(e, path))


def flip_and_save(image_ori,filename_target):
    print("Flip image:" + filename_target)
    image_fliped = image_ori.transpose(Image.FLIP_LEFT_RIGHT)
    image_fliped.save(filename_target, 'JPEG', quality=100, optimize=True, progressive=True)


def load_dir_and_extract_faces(src_dir, target_dir):
    for subdir in listdir(src_dir):
        path = src_dir + subdir + FILEPATH_DIVISOR
        path_target = target_dir + subdir + FILEPATH_DIVISOR

        if not isdir(path):
            continue

        load_photos(path, path_target)


##################################
# MAIN
##################################
if __name__ == '__main__':
    print("############################################")
    print("Summary of files into dirs")
    summary_files(PHOTOS_DIR)
    summary_files(FACES_DIR)

    print("############################################")
    print("init extract faces")
    load_dir_and_extract_faces(PHOTOS_DIR, FACES_DIR)

    print("############################################")
    print("summary after extract faces")
    summary_files(FACES_DIR)


