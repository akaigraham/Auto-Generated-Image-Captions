# import dependencies
from array import array
import numpy as np
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.preprocessing import sequence, image 
from keras import Input, layers, optimizers
from keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout
from keras.layers.merge import add
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from typing import NewType

# preprocess image for transfer learning
# with InceptionV3
def preprocess(image_path, resize=(350,350)):
    """
    Preprocesses image using InceptionV3. 
    Input: 
        image_path: sring denoting path to image file
        resize: tuple of ints dentoing size to resize image to
    Returns: 
        numpy.array or a tf.Tensor with type float32
    """

    # load and convert image to numpy array
    img_pil = image.load_img(image_path, target_size=resize)
    img_numpy = image.img_to_array(img_pil)

    # expand shape of array
    img_expanded = np.expand_dims(img_numpy, axis=0)

    # preprocess for InceptionV3 and return
    return preprocess_input(img_expanded)

# encode preprocessed image using InceptionV3
def encode(image_path):
    """
    Instantiates InceptionV3 for transfer learning. Passes preprocessed image 
    file to InceptionV3 model.  
    Input: 
        image_path: string denoting path to image file
    Returns:
        [...]
    """

    # instantiate V3 model and set weights
    transfer_model = InceptionV3(weights='imagenet')

    # remove softmax layer as not needed for image classification
    new_transfer_model = Model(transfer_model.input, 
                               transfer_model.layers[-2].output)

    # preprocess and generate predictions
    img_preprocessed = resize(image_path)
    img_preds = new_transfer_model.predict(img_preprocessed)

    # reshape and return
    return np.reshape(img_preds, img_preds.shape[1])

# function to set up model
def generate_caption(image_path, in_text='startseq', max_length=80,
                     word_to_idx, idx_to_word, model):
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
        y_hat = model.predict([image_path, sequence], verbose=0)
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



# Build Model
def build_model(input_shape, max_length, embedding_dim, vocab_size):
    """
    Build model that will be used to auto-generate captions.
    """

    # create model
    inputs_1 = Input(shape=(input_shape,))
    fe1 = Dropout(0.5)(inputs_1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs_2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs_2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # create decoder
    decoder_1 = add([fe2, se3])
    decoder_2 = Dense(256, activation='relu')(decoder_1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # return model
    return Model(inputs=[inputs_1, inputs_2], outputs=outputs)


