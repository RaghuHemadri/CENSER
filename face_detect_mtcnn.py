import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import load_model
from keras_vggface.utils import preprocess_input
from glob import glob
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from PIL import Image
import cv2

# create the detector, using default weights
detector = MTCNN()

def extract_face(filename, required_size=(160, 160), detector = detector):
    # load image from file
    pixels = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    pixels = cv2.copyMakeBorder(pixels, 100, 100, 100, 100, cv2.BORDER_REPLICATE)
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    if len(faces) == 0:
        return 0
    else:
        x1, y1, width, height = faces[0]['box']
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        img = Image.fromarray(face)
        img = img.resize(required_size)
        return img

if __name__ == "__main__":
	# Read Image
        train_folders_path = 'old_new_face/'
        all_images = glob(train_folders_path + "*/*.jpg")
        all_images += glob(train_folders_path + "*/*.png")
        all_images += glob(train_folders_path + "*/*.PNG")

        for im in all_images:
            cropped_im = extract_face(im)
            if cropped_im == 0:
                continue
            else:
                cropped_im.save(im)
