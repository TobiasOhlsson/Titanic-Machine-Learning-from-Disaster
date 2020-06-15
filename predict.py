from read_data import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

model_name = "FeedForward"

data = read_data('test')
X_test = data.to_numpy()
model = load_model('Models/' + model_name + '.hdf5')
predictions = model.predict(X_test)
predictions = np.around(predictions).flatten()
predictions = predictions.astype(int)

d = {'PassengerId': range(892, 1310), 'Survived': predictions}
df = pd.DataFrame(data=d)
df.to_csv('Results/' + model_name + '.csv', index=False)
