from feature_extraction_net import feature_extraction_net
from imggen import img_gen

def feature_extraction(input_shape, directory):
    feature_extraction = feature_extraction_net(input_shape)
    images = img_gen(directory, 1)
    features = feature_extraction.predict(images, steps=10000)
    return features