from keras.layers import Input, Dense
from keras.models import Model
from keras.regularizers import l2
import numpy as np


def get_autoenc(X, hidden, optimizer='adadelta', loss='binary_crossentropy'):
    """ Stacked Autoencoder using Keras framework with Theano Backend

        The function takes in the training data, number of hidden layers &
        neurons, and optionally the optimizer algorithm and loss function.

        default loss function: Binary Cross Entropy
        default optimizer: Adadelta optimizer (Stochastic Gradient Descent)

        The function return a keras.Model object that acts as the encoder part
        of the autoencoder structure.

        L2 Weight Regularizer to prevent over-fitting: lambda = 1

        (numpy.array, [int], <str>, <str>) -> (keras.Model)

        References:
        http://sebastianruder.com/optimizing-gradient-descent/index.html#adadelta
        http://blog.keras.io/building-autoencoders-in-keras.html
        http://keras.io/models/model/
    """
    M = X.shape[1]
    input_img = Input(shape=(M,))

    # Input layer
    encoded = Dense(M, init='uniform', activation='relu',
                    W_regularizer=l2(1))(input_img)

    # Start stacking hidden layers
    for h in hidden:
        encoded = Dense(h, init='uniform', activation='relu',
                        W_regularizer=l2(1))(encoded)

    decoded = Dense(h, init='uniform', activation='relu',
                    W_regularizer=l2(1))(encoded)
    for h in hidden[-2::-1]:
        decoded = Dense(h, init='uniform', activation='relu',
                        W_regularizer=l2(1))(decoded)
    decoded = Dense(M, init='uniform', activation='softmax')(decoded)

    encoder = Model(input=input_img, output=encoded)
    autoencoder = Model(input=input_img, output=decoded)
    encoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.compile(optimizer=optimizer, loss=loss)

    return (encoder, autoencoder)


def encode(c):
    """ Encoding scheme for the input statespace, using uniformly spaced values
        with considerable gap to prevent autoencoder approximation from
        overlapping with other symbol values.

        (chr) -> (int)
    """
    if c == '%':
        return 50
    elif c == ' ':
        return 100
    elif c == '.':
        return 250
    elif c == 'G':
        return 300
    elif c == 'o':
        return 350
    elif c in ['>', '<', 'v', '^']:
        return 400


def normalize(encoded_imgs, a=50, b=400):
    """ Normalize the input array between two values. The boundaries default
        to (50, 400), the upper and lower bounds of the encoding scheme.

        (numpy.array, <num>, <num>) -> (numpy.array)
    """
    return a + (((encoded_imgs - encoded_imgs.min())
                 * (b - a)) / (encoded_imgs.max() - encoded_imgs.min()))


def train(train_data='state_file_4_conv_uniq.dat', epoch=50, hidden=[50]):
    """ Train the autoencoder for a specified number of epochs with a given
        hidden layer structure on the provided dataset.

        NOTE: the train function is tightly coupled with the autoencoder and
        the pacman state-space.

        (np.array, int, [int]) -> keras.Model
    """

    # get saved possible states and encode them.
    dataset = []
    with open(train_data, 'r') as orig:
        rows = []
        for l in orig:
            if l == '\n':
                dataset.append(rows)
                rows = []
                continue
            row = []
            l = l.rstrip('\n')
            if l.count('%') == len(l):
                continue
            else:
                l = l[1:-1]
            for c in list(l):
                row.append(encode(c))
            rows.append(row)

    x_train = np.array(dataset)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    # get Autoencoder
    encoder, autoencoder = get_autoenc(x_train, hidden)

    # Split into validation(20%) and training set
    val = int(np.ceil(0.2 * x_train.shape[0]))
    x_train_sub = x_train[:len(x_train) - val]
    x_test = x_train[-val:]

    # Normalize the input
    x_train_sub_norm = normalize(x_train_sub)
    x_test_norm = normalize(x_test)

    # Train the autoencoder in the training set
    autoencoder.fit(x_train_sub_norm, x_train_sub_norm,
                    nb_epoch=epoch,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_norm, x_test_norm))

    return encoder


def predict(encoder, state):
    """ Using the provided encoder return the compressed state for the given state.

        (keras.Model, GameState) -> (tuple)
    """
    # Stringify the GameState
    str_state = str(state).split('Score')[0].split('\n')

    # transform the state into [int] using encoding scheme
    dataset = []
    rows = []
    for l in str_state:
        if l == '':
            continue
        row = []
        l = l.rstrip('\n')
        if l.count('%') == len(l):
            continue
        else:
            l = l[1:-1]
        for c in list(l):
            row.append(encode(c))
        rows.append(row)
    dataset.append(rows)

    # Using the encoder part of the trained autoencoder, predict
    # the encoded state.
    x_train = np.array(dataset)
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    state_encoded = encoder.predict(x_train)
    return tuple(list(np.round(normalize(state_encoded)[0])))

if __name__ == '__main__':
    train(hidden=[50, 20])
