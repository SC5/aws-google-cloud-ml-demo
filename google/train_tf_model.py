import csv
import shutil
import numpy as np
import keras
import keras.backend as K

# np.random.seed(1337) # Seed the random number generator for consistent results

from keras.models import Sequential 
from keras import regularizers
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical

# Import TensorFlow model saving utilities
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.contrib.session_bundle import exporter

X = []
Y = []

# Create training data (features, X and labels, Y)
with open('../datasets/iris/Iris.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  next(reader, None)  # Skip header
  for row in reader:
    X.append([
      float(row[1]), # SepalLengthCm
      float(row[2]), # SepalWidthCm
      float(row[3]), # PetalLengthCm
      float(row[4])  # PetalWidthCm
    ])
    if row[5] == 'Iris-setosa':
      Y.append(0)
    elif row[5] == 'Iris-versicolor':
      Y.append(1)
    else:
      Y.append(2)
X = np.array(X)

# Convert labels to one-hot vectors (required by Keras):
# Label 0 becomes [1,0,0]
# Label 1 becomes [0,0,1]
# Label 2 becomes [0,0,1]
Y = to_categorical(Y)

# Create the logistic regression (classification) model
model = Sequential()
#model.add(Dense(3, activation='relu', input_dim=X.shape[1]))
model.add(Dense(3, activation='softmax', input_dim=X.shape[1]))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=10000, shuffle=True, validation_split=0.2)

print('Iris-setosa has class number 0')
print('Iris-versicolor has class number 1')
print('Iris-virginica has class number 2')

print('Sample prediction using features [5.6,2.8,4.9,2.0] (should be Iris-virginica):')
test_example = np.array([[5.6,2.8,4.9,2.0]])
print('SepalLengthCm: 5.6')
print('SepalWidthCm: 2.8')
print('PetalLengthCm: 4.9')
print('PetalWidthCm: 2.0')
print('Predicted class: ' + str(np.argmax(model.predict(test_example)[0])))
print('Prediction distribution: ' + str(model.predict(test_example)[0]))

shutil.rmtree('model/')

# Save TensorFlow compatible model
builder = saved_model_builder.SavedModelBuilder('model/')
signature = predict_signature_def(inputs={'features': model.input},
                                  outputs={'classes': model.output})
with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={'serving_default': signature})
    builder.save() # Save model to export_path
