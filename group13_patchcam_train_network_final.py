import gzip
import h5py
import tensorflow as tf
import numpy as np
import keras
from pathlib import Path
import time
import gc

path = r"/scratch-local/sdvries/sdvries"

# set seed
np.random.seed(0)
tf.random.set_seed(0)

# Training data
fx_train = gzip.open(path + r'/camelyonpatch_level_2_split_train_x.h5.gz','rb')
fy_train = gzip.open(path + r'/camelyonpatch_level_2_split_train_y.h5.gz','rb')

x_train_h5 = h5py.File(fx_train, 'r')
x_train = x_train_h5['x'][:]
x_train_h5.close()

y_train_h5 = h5py.File(fy_train, 'r')
y_train = y_train_h5['y'][:]
y_train_h5.close()

# One hot encode y
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_train_one_hot =keras.utils.np_utils.to_categorical(y_train, 2)


# Validation data
fx_validation = gzip.open(path + r'/camelyonpatch_level_2_split_valid_x.h5.gz','rb')
fy_validation = gzip.open(path + r'/camelyonpatch_level_2_split_valid_y.h5.gz','rb')

x_validation_h5 = h5py.File(fx_validation, 'r')
x_validation = x_validation_h5['x'][:]
x_validation_h5.close()

y_validation_h5 = h5py.File(fy_validation, 'r')
y_validation = y_validation_h5['y'][:]
y_validation_h5.close()

# One hot encode y
y_validation = np.asarray(y_validation).astype('float32').reshape((-1,1))
y_validation_one_hot =keras.utils.np_utils.to_categorical(y_validation, 2)

def train_network(network: keras.Model,
                x_training: np.ndarray, 
                y_training: np.ndarray, 
                x_validation: np.ndarray, 
                y_validation: np.ndarray, 
                n_epoch: int,
                batch_size: int, 
                network_filepath: Path,
                class_weights: dict = None,
                verbose: int = 1,
                save: bool = True,
                ):

    best_validation_accuracy = 0

    for epoch in range(n_epoch):
        print("Epoch: ", epoch)
        st = time.time()

        # Train the network
        results = network.fit(x_training, y_training, batch_size = batch_size, shuffle=True, class_weight = class_weights, verbose = verbose)

        # Evaluate performance (loss and accuracy) on validation set
        scores = network.evaluate(x_validation, y_validation, batch_size = batch_size, verbose = verbose)
        
        validation_accuracy = scores[1]

        # Save model if it has better validation accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            if save:
                try:
                    network.save(network_filepath + ".h5")
                    print("saved the model to {}".format(network_filepath))
                except:
                    print("Unable to save the model")
                    pass

        gc.collect()

def hard_negative_gen(network, x_train, y_train):
    # Get all negative cases
    mask_true_neg = (y_train[:,0] == 1)
    x_neg = x_train[mask_true_neg]
    y_neg = y_train[mask_true_neg]

    # Get all positive cases
    mask_true_pos = (y_train[:,1] == 1)
    x_pos = x_train[mask_true_pos]
    y_pos = y_train[mask_true_pos]

    # Predict label for all true negative training cases
    predictions = network.predict(x_neg)
    
    # Argsort predictions
    sort_pred_args = np.argsort(predictions[:,0])

    # Remove 10% of the easiest negative cases
    rem_percent = x_neg.shape[0]/7
    keep_index = int(np.floor(x_neg.shape[0] - rem_percent))
    x_neg_new = x_neg[sort_pred_args[0:keep_index]]
    y_neg_new = y_neg[sort_pred_args[0:keep_index]]

    # Create new data
    x_train = np.concatenate((x_neg_new, x_pos))
    y_train = np.concatenate((y_neg_new, y_pos))

    # Update class weights
    class_weights = {0: x_neg_new.shape[0]/x_neg_new.shape[0], 1: x_neg_new.shape[0]/x_pos.shape[0]}
    
    return x_train, y_train, class_weights


def train_hnm_network(model_name, x_train, x_validation, y_train, y_validation, nr_gen=3, n_epoch=40, batch_size=24):
    class_weights = {0: 1, 1: 1 }

    for i in range(0, nr_gen):
        # Set up parameters
        loss =  tf.keras.losses.CategoricalCrossentropy(from_logits = False)
        learning_rate = 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

        # Data augmentation
        inputs = tf.keras.Input(shape=(None, None, 3))
        processed_1 = keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=0)(inputs)
        processed_2= keras.layers.RandomRotation((-0.5, 0.5), fill_mode="reflect", interpolation="bilinear", seed=0, fill_value=0.0)(processed_1)
        processed_3 = keras.layers.RandomZoom(height_factor = 0.2, width_factor = 0.2, fill_mode = "reflect", interpolation = "bilinear", seed  = 0)(processed_2)

        # Initialize pre_built network
        if model_name == 'EfficientNetB3':
            outputs = tf.keras.applications.EfficientNetB3(weights=None, classes=2, classifier_activation='sigmoid', input_shape=(96, 96, 3))(processed_3)
        elif model_name == 'InceptionV3':
            outputs = tf.keras.applications.InceptionV3(weights=None, classes=2, classifier_activation='sigmoid', input_shape=(96, 96, 3))(processed_3)
        elif model_name == 'DenseNet201':
            outputs = tf.keras.applications.DenseNet201(weights=None, classes=2, classifier_activation='softmax', input_shape=(96, 96, 3))(processed_3)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        network_filepath = f'/home/sdvries/models/efficient/{model_name}_hnm_gen_{i}'

        # Train network
        train_network(model, x_train, y_train, x_validation, y_validation, n_epoch, batch_size, network_filepath, save=True, class_weights = class_weights, verbose = 1)
        # Adjust data through hard negative mining
        x_train_full, y_train_one_hot_full, class_weights = hard_negative_gen(model, x_train_full, y_train_one_hot_full)
        print("training has ended for model ", str(i))

model_names = ['EfficientNetB3','InceptionV3','DenseNet201']
train_hnm_network(model_names[0], x_train, x_validation, y_train_one_hot, y_validation_one_hot, nr_gen=5, n_epoch=15, batch_size=24)
train_hnm_network(model_names[1], x_train, x_validation, y_train_one_hot, y_validation_one_hot, nr_gen=5, n_epoch=15, batch_size=24)
train_hnm_network(model_names[2], x_train, x_validation, y_train_one_hot, y_validation_one_hot, nr_gen=3, n_epoch=40, batch_size=24)

