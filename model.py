# import dependencies
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import sequence, image 
from keras import Input, layers, optimizers
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical





def resize(image_path, new_size=(299,299)):
    """
    Reshape input img to new_size
    Input:
        image_path: string designating path to image file
        new_size: tuple denoting output size. Defaults to (299 x 299)
    """
    # load image into PIL format
    img = image.load_img(image_path, target_size=new_size)

    # convert PIL image to Numpy array
    X = image.img_to_array(img)

    # expand shape of array
    X = np.expand_dims(X, axis=0)

    # preprocess and return
    return preprocess_input(X)

def encode(image_path):
    """
    Resizes image and generates a feature vector using InceptionV3
    """

    # instantiate V3 model
    transfer_model = InceptionV3(weights='imagenet')

    # remove softmax layer - not needed for image classification
    new_transfer_model = Model(transfer_model.input, 
                               transfer_model.layers[-2].output)

    # resize img
    img_resized = resize(image_path)

    # vectorize image
    img_vect = new_transfer_model.predict(img_resized)

def generate_captions(image_path, search_type='greedy', k=3):
    """
    Encodes image file found at image_path using InceptionV3. 
    Encoded image passed to trained model and returns generated captions using 
    either greedy search or beam search. 

    Input:
        image_path: path to image
        search_type: string denoting search type (greedy or beam)
        k: number of neighbors to use if search_type==beam

    Returns:
        Caption describing input photo.
    """
    
    return None
