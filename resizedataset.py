import tensorflow as tf
import os
from zipfile import ZipFile
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


def resize_dataset(zip_file, dirpath, img_shape):
    zipref = ZipFile(zip_file, 'r')
    zipref.extractall()
    directory = zipref.filename[:-4]
    zipref.close()

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    count = 0
    size = len(os.listdir(directory))
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = load_img(directory+"/"+filename)
            img = img_to_array(img)
            img = tf.image.resize_with_pad(img, img_shape[0], img_shape[1], antialias=True)
            img = array_to_img(img)
            img.save(dirpath+"/"+str(int(os.path.splitext(filename)[0]))+".jpg")
