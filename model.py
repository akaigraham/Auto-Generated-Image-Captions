# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup 
from PIL import Image
from nltk import word_tokenize
import re

# transfer learning model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import sequence, image 

# progress bars
from alive_progress import alive_bar; import time

# serialization
import pickle 

# deep learning model
from keras import Input, layers, optimizers
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import load_model


# instantiate V3 model and set weights
transfer_model = InceptionV3(weights='imagenet')

# remove softmax layer as not needed for image classification  
new_transfer_model = Model(transfer_model.input, transfer_model.layers[-2].output)

# preprocess image for transfer learning
# with InceptionV3
def resize(image_path):
    """
    Preprocesses image using InceptionV3. 
    Input: 
        image_path: sring denoting path to image file
        resize: tuple of ints dentoing size to resize image to
    Returns: 
        numpy.array or a tf.Tensor with type float32
    """

    # load and convert image to numpy array
    img_pil = image.load_img(image_path, target_size=(299,299))
    X = image.img_to_array(img_pil)

    # expand shape of array
    X = np.expand_dims(X, axis=0)

    # preprocess for InceptionV3 and return
    return preprocess_input(X)


# encode preprocessed image using InceptionV3
def encode(image_path):
    """
    Instantiates InceptionV3 for transfer learning. Passes preprocessed image 
    file to InceptionV3 model.  
    Input: 
        image_path: string denoting path to image file
    Returns:
        Encoded image.
    """

    # preprocess and generate predictions
    img_preprocessed = resize(image_path)
    img_preds = new_transfer_model.predict(img_preprocessed)

    # reshape and return
    return np.reshape(img_preds, img_preds.shape[1])

# function to set up model
def generate_caption(encoded_image, word_to_idx, idx_to_word, 
                     model, in_text='startseq', max_length=80):
    """
    Use greedy search and cloud_trained model to generate captions.
    Input:
        image_path: string denoting path to image file
        in_text: text denoting a start sequence, defaults to startseq
        max_length: max_length of caption
        word_to_idx: dictionary of word to index key value pairs
        idx_to_word: dictionary of index to word key value pairs
        model: model used to generate predictions
    Returns: 
        string description of image contents
    """
    # build caption starting with startseq
    for i in range(max_length):

        # set up sequence: get index for each word in in_text
        # check that the word is in word_to_idx
        # pad sequences so all the same length
        sequence = [word_to_idx[j] for j in in_text.split() if j in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_length)

        # use model to generate predictions 
        # get highest probability prediction for next word
        y_hat = model.predict([encoded_image, sequence], verbose=0)
        y_hat = np.argmax(y_hat)

        # convert model prediction back to string
        # append word to in_text
        word = idx_to_word[y_hat]
        in_text += ' ' + word

        # if finished building caption, break
        if word == 'endseq':
            break
        
    # return caption
    caption_tokens = in_text.split() # split in_text by word
    caption_tokens = caption_tokens[1:-1] # ignore startseq and endseq
    return ' '.join(caption_tokens) # return string caption
    

def main():
    # Load model and necessary objects
    trained_model = load_model('cloud_trained_model.h5')

    # load word_to_idx
    with open('cloud_files/word_to_idx.pickle', 'rb') as f:
        word_to_idx = pickle.load(f)

    # load idx_to_word
    with open('cloud_files/idx_to_word.pickle', 'rb') as f:
        idx_to_word = pickle.load(f)

    # test on image not in test set
    test_img_path = '/Users/addingtongraham/Desktop/running_dog.jpg'

    # encode img using InceptionV3
    test_img_encoded = encode(test_img_path).reshape((1,2048))

    # generate captions
    caption = generate_caption(test_img_encoded, word_to_idx=word_to_idx, idx_to_word=idx_to_word, model=trained_model)

    print('test')
    print(f'Generated Caption:\n{caption}')

if __name__ == '__main__':
    main()
