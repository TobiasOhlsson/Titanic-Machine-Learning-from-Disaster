from read_data import *
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# this script builds and trains a simple feed forward neural network using the Keras api of tensorflow

data = read_data('train')
y = data["Survived"].to_numpy()
X = data.drop(columns="Survived").to_numpy()

# Definition of the networks hyper-parameters
input_dim = len(X[0])
epochs = 100
hidden_sizes = [10, 20, 10]

# Definition of the model

input_placeholder = Input(shape=(input_dim,))
layer = input_placeholder
while len(hidden_sizes) > 0:
    dim = hidden_sizes.pop(0)
    layer = Dense(dim, activation='sigmoid')(layer)
    layer = Dropout(0.1)(layer)
output = Dense(1, activation='sigmoid')(layer)
model = Model(input_placeholder, output)

# Compile model
opt = Adam(lr=1e-2, decay=1e-2/epochs)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training of the model
model.fit(X, y,
          epochs=epochs,
          batch_size=20,
          shuffle=True)
# save the model
model.save("Models/FeedForward.hdf5")
