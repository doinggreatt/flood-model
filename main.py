import numpy as np 
import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt  

def print_intro():
    print('[I] This is a CLI application to use and validate water level forecasting neural networks')
    print('[I] They are based on Conditional Model Selection. There are 4 water bodies available.')
    print('[I] 1. р. Абылайкит - с. Самсоновка (Усть-Каменогорск), 2. р. Буктырма - c. Лесная Пристань (Алтайский район)')
    print('[I] 3. р. Аксу - с. Аксу (Катон-Карагайский район), 4. р. Куршим - c. Вознесенка (Курчумский район)')
    print('----------------------------')
    print('[I] Forecast is given a week ahead, based on previous 7 days. 5 inner features are used.')
    print('[I] They are: Partial pressure (Парциальное давление); Atmospheric precipitation (Кол-во атмосферных осадков);')
    print('[I] Water levels (Уровни воды); Water consumption (Расход воды); Seasonality (Сезонность)')
    print('----------------------------')
    print('[I] The average raw data MSE of all models: ~45-53.')
    print('[I] Now you have 2 available options.')
    print('[I] [1] Validate data on the validation sample.')
    print('[I] [2] Make a real prediction.')

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
    
def validate_data_intro():
    print('[I] Please, choose the water boddy.')
    print('[1] р. Абылайкит - с. Самсоновка (Усть-Каменогорск)')
    print('[2] р. Буктырма - c. Лесная Пристань (Алтайский район)')
    print('[3] р. Аксу - с. Аксу (Катон-Карагайский район)')
    print('[4] р. Куршим - c. Вознесенка (Курчумский район)')
    try: 
        num = int(input('[I] Choose an option: '))
        if num == 1 or num == 2 or num == 3 or num == 4:
            validate_data(num)
        else:
            print('[E] Invalid option selected. Please choose 1, 2, 3, or 4.')
    except ValueError:
            print('[E] Invalid input. Please enter a number.')


def create_time_steps(length):
  return list(range(-length, 0))


def model_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/1, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


def validate_data(num):
    plot_n = int(input('How many plots you want to draw? '))
    codes = { 
      1: '11661.h5',
      2: '11129.h5',
      3: '11143.h5',
      4: '11146.h5',
    }
    splits = {  # Кортеж для отделения валидационной выборки от обучающей. (TRAIN_SPLIT)
        1: 4100,
        2: 3300, 
        3: 5100,
        4: 7000,
    }
    dataset = dataset_prepare(num)
    split = splits[num]
    _model = 'models/model-' + codes[num]
    model = tf.keras.models.load_model(_model, compile=False)
    model.compile(optimizer='adam', loss='mse')
    
    x_val, y_val = multivariate_data(dataset, dataset[:, 0], split , None, 7, 7, 1)

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.cache().batch(64).repeat()
    
    val_loss = model.evaluate(val_data, steps=50)
    print('Loss value:',val_loss)
    print('Plotting graphs...')
    for x,y in val_data.take(plot_n):
        pred = model.predict(x)[0]

        model_plot(x[0], y[0], pred)



def get_user_predictions():
    print("Введите значения для следующих признаков:")
    data = []
    for i in range(1, 8):
        water_level = float(input(f"Уровень воды в {i} день: "))
        water_cons = float(input(f"Расходы воды в {i} день: "))
        parc_press = float(input(f"Парциальное давление в {i} день: "))
        osadki = float(input(f"Кол-во суточных атмосферных осадков в {i} день: "))
        data.append([water_level, water_cons, parc_press, osadki])
    _month = float(input("Меcяц(от 1 до 12): "))
    month =  np.sin(2 * np.pi * _month / 12)
    data = np.array(data)
    data = np.hstack([data, np.full((7,1), month)])
    data = np.expand_dims(data, axis=-1)



    return data

def standardize_data(data, mean, std):
    return (data - mean) / std 

def destandardize_data(pred, mean, std):
    return 0
def make_prediction(user_data, num):
    codes = { 
      1: '11661.h5',
      2: '11129.h5',
      3: '11143.h5',
      4: '11146.h5',
    }
    splits = {  # Кортеж для отделения валидационной выборки от обучающей. (TRAIN_SPLIT)
        1: 4100,
        2: 3300, 
        3: 5100,
        4: 7000,
    }
    model = tf.keras.models.load_model('models/model-'+codes[num], compile=False)
    model.compile(optimizer='adam', loss='mse')
    pred = model.predict(user_data)
    return pred
def make_prediction_intro():
    
    _user_data = get_user_predictions()

    print('[I] Please, choose the water boddy.')
    print('[1] р. Абылайкит - с. Самсоновка (Усть-Каменогорск)')
    print('[2] р. Буктырма - c. Лесная Пристань (Алтайский район)')
    print('[3] р. Аксу - с. Аксу (Катон-Карагайский район)')
    print('[4] р. Куршим - c. Вознесенка (Курчумский район)')
    try: 
        num = int(input('[I] Choose an option: '))
        if num == 1 or num == 2 or num == 3 or num == 4:
            data_mean, data_std =  dataset_prepare(num, True)
            user_data = standardize_data(_user_data, data_mean, data_std)
            pred = make_prediction(user_data, num)
            # pred_raw = destandardize_data(pred, data_mean, data_std)
            # for i in pred_raw:
            # print(i)
            pred_raw = pred * data_std[0] + data_mean[0]
            for i in pred_raw[0]:
                print(i)
        else:
            print('[E] Invalid option selected. Please choose 1, 2, 3, or Q.')
    except ValueError as _ex:
            print('[E] Invalid input. Please enter a number.', _ex)


def dataset_prepare(num, need_mean_data =False):
    paths = { 
        1: '11661',
        2: '11129', 
        3: '11143', 
        4: '11146'
    }
    file_path = 'datasets/' + paths[num] +'.xlsx'
    df = pd.read_excel(file_path)

    df['month'] = df['date'].dt.month

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    features_considered = ['water_level', 'water_cons', 'month_sin', 'parc_press', 'osadki']
    features = df[features_considered] 
    _features = features.copy()
    _features.fillna(features.mean(), inplace=True) # Avoiding NaN meanings in the sample

    dataset = _features.values

    data_mean = dataset.mean(axis=0)
    data_std = dataset.std(axis=0)

    dataset=(dataset-data_mean)/data_std
    if need_mean_data:
        return data_mean, data_std
    return dataset

def main():
    print_intro()
    try:
        opt = int(input('[I] Choose an option: '))
        if opt == 1:
            validate_data_intro()
        elif opt == 2:
            make_prediction_intro()
        else:
            print('[E] Invalid option selected. Please choose 1 or 2.')
            
    except ValueError:
        print('[E] Invalid input. Please enter a number.')

if __name__ == '__main__':
    main()