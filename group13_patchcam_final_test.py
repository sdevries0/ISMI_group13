import gzip
import h5py
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras import metrics
from pathlib import Path
import pandas as pd
from datetime import datetime

path = r"/scratch-local/sdvries/sdvries"

# set seed
np.random.seed(0)
tf.random.set_seed(0)

# Test data 
fx_test = gzip.open(path + r'/camelyonpatch_level_2_split_test_x.h5.gz','rb')
fy_test = gzip.open(path + r'/camelyonpatch_level_2_split_test_y.h5.gz','rb')

x_test_h5 = h5py.File(fx_test, 'r')
x_test_full = x_test_h5['x'][:]
x_test_h5.close()

y_test_h5 = h5py.File(fy_test, 'r')
y_test_full = y_test_h5['y'][:]
y_test_h5.close()
# One hot encode y
y_test_full = np.asarray(y_test_full).astype('float32').reshape((-1,1))
y_test_one_hot_full =keras.utils.np_utils.to_categorical(y_test_full, 2)

def TTA(X, model):
    # Augmentation layers
    a1 = keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=0)
    a2 = keras.layers.RandomRotation((-0.5, 0.5), fill_mode="reflect", interpolation="bilinear", seed=0, fill_value=0.0)
    a3 = keras.layers.RandomContrast(0.2, seed=0)
    a4 = keras.layers.RandomZoom(height_factor = 0.2, width_factor=0.2,fill_mode="reflect",interpolation="bilinear",seed=0)

    # Predict each augmentated image
    p0 = model.predict(X)
    p1 = model.predict(a1(X, training = True))
    p2 = model.predict(a2(X, training = True))
    p3 = model.predict(a3(X, training = True))
    p4 = model.predict(a4(X, training = True))

    # Average predictions
    p = (p0 + p1 + p2 + p3 + p4) / 5
    return np.array(p)

model_names = ['inception','efficient','densenet']
nr_gen = 5
models=[]
for name in model_names:
    # DenseNet only has three generations
    if name == 'densenet':
        nr_gen = 3

    # Load checkpoint for every generation
    for gen in range(0, nr_gen):
        file_dir = f'/home/sdvries/models/{name}/'
        network_filepath = Path(file_dir, 'hnm_gen_' + str(gen)+'.h5')
        model = keras.models.load_model(network_filepath)
        model._name = "hnmnet" + str(gen) + name
        models.append(model)

# To compile the model
loss =  tf.keras.losses.CategoricalCrossentropy(from_logits = False)
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
metrics = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

# Create ensemble model
model_input = tf.keras.Input(shape=(96, 96, 3))
model_outputs = [model(model_input) for model in models]
ensemble_output = tf.keras.layers.Average()(model_outputs)
ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

ensemble_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# With TTA
predictions = TTA(x_test_full,ensemble_model)

# Create a csv with predictions and true labels
df = pd.DataFrame(predictions[:,1], columns=['prediction'])
df.index.name = 'case'

csv_filename = '/home/sdvries/predictions/predictions_TTA.csv'
df.to_csv(csv_filename)

# Without TTA
predictions = ensemble_model.predict(x_test_full)

# Create a csv with predictions and true labels
df = pd.DataFrame(predictions[:,1], columns=['prediction'])
df.index.name = 'case'

date = str(datetime.now())
csv_filename = '/home/sdvries/predictions/predictions.csv'

df.to_csv(csv_filename)

