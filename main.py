import numpy as np
import tensorflow as tf
import os
import pickle
import random

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from resizedataset import resize_dataset
from feature_extraction import feature_extraction
from build_tensor import build_tensor

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)
IMAGE_SHAPE = [299,299]
INPUT_SHAPE = (299,299,3)
NUM_VALIDATION_TRIPLETS = 1500
NUM_TRAINING_IMAGES = 5000
ARCHIVE = "food.zip"
RESIZED_IMAGE_DIRECTORY = "./food_res/"
FEATURES_FILE = "features.pckl"
TRAIN_FILE = "train_triplets.txt"
TEST_FILE = "test_triplets.txt"
SUBMISSION_FILE = "submission.txt"
MODEL_FILE = "model.keras"


def main():
    # print("RESIZING IMAGES...")
    # resize_dataset(ARCHIVE, RESIZED_IMAGE_DIRECTORY, IMAGE_SHAPE)
    # print("RESIZING IMAGES DONE.")
    if os.path.exists(FEATURES_FILE):
        with open(FEATURES_FILE, 'rb') as f:
            features = pickle.load(f)
    else:
        print("EXTRACTING FEATURES WITH PRETRAINED VISUAL MODEL")
        features = feature_extraction(INPUT_SHAPE, RESIZED_IMAGE_DIRECTORY)
        with open(FEATURES_FILE, 'wb') as f:
            pickle.dump(features, f)

    print("FEATURES LOADED.")
    print("BUILDING TRAINING AND TEST TENSORS...")
    train_tensors, labels = build_tensor(features, "train_triplets.txt", gen_labels=True)
    test_tensors = build_tensor(features, "test_triplets.txt", gen_labels=False)
    print("TENSORS GENERATED")


    print("BUILDING MODEL...")
    x = x_in = Input(train_tensors.shape[1:])
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1152)(x)
    x = Activation('relu')(x)
    x = Dense(288)(x)
    x = Activation('relu')(x)
    x = Dense(72)(x)
    x = Activation('relu')(x)
    x = Dense(18)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=x_in, outputs=x)
    print("BUILDING MODEL DONE.")
    print("COMPILING MODEL...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("COMPILING MODEL DONE.")

    if not os.path.exists(MODEL_FILE):
        print("TRAINING MODEL...")
        model.fit(x=train_tensors, y=labels, epochs=35)
        model.save(MODEL_FILE)
        print("TRAINING MODEL DONE.")
    else:
        model = tf.keras.models.load_model(MODEL_FILE)

    print("PREDICTING RESULT...")
    prediction = model.predict(test_tensors)
    print("PREDICTION DONE.")

    print("CREATING SUBMISSION..")
    threshold = np.where(prediction < 0.5, 0, 1)
    np.savetxt(SUBMISSION_FILE, threshold, fmt='%d')
    print("SUBMISSION FILE GENERATED.")


if __name__ == '__main__':
    main()