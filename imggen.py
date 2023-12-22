import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def img_gen(dirpath, batch_size):
    index = 0
    while True:
        batch = []
        while len(batch) < batch_size:
            img_name = dirpath+str(int(index))+".jpg"
            img = load_img(img_name)
            img = tf.keras.applications.inception_resnet_v2.preprocess_input(img_to_array(img))
            batch.append(img)
            index = (index+1)%10000

        batch = np.array(batch)
        labels = np.zeros(batch_size)

        try: 
            yield batch, labels
        except StopIteration:
            return
