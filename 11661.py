import tensorflow as tf 
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_squared_error


df = pd.read_excel('datasets/11661.xlsx')

tf.random.set_seed(13)

df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)


features_considered = ['water_level', 'water_cons', 'month_sin', 'parc_press', 'osadki']
features = df[features_considered]
TRAIN_SPLIT = 4100

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset=(dataset-data_mean)/data_std
EPOCHS = 40
EVALUATION_INTERVAL = 120
past_history = 7
future_target = 7
STEP = 1
VAL_STEPS =18
BATCH_SIZE = 64
BUFFER_SIZE = 1000  

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


x_train, y_train = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP)
x_val, y_val = multivariate_data(dataset, dataset[:, 0], TRAIN_SPLIT, None, past_history, future_target, STEP)
print(x_val.shape)
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.cache().batch(BATCH_SIZE).repeat()
model = tf.keras.models.Sequential() 

model.add(tf.keras.layers.LSTM(5, input_shape=x_train.shape[-2:], return_sequences=True))
model.add(tf.keras.layers.LSTM(10, activation='relu'))
model.add(tf.keras.layers.Dense(future_target))

model.compile(optimizer='adam', loss='mse') 

model_history = model.fit(
    train_data, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data, validation_steps = VAL_STEPS)




val_predictions = model.predict(x_val)
val_predictions_raw = val_predictions * data_std[0] + data_mean[0]
y_val_raw = y_val * data_std[0] + data_mean[0]
mse = mean_squared_error(y_val_raw, val_predictions_raw)
val_loss = model.evaluate(val_data, steps=VAL_STEPS)
print('Raw data mse:', mse)
print('MSE:', val_loss)

model.save('models/model-11661.h5')
