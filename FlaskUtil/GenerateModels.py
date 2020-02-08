import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cdist
import operator
from smart_open import smart_open
import os
from configparser import ConfigParser
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt

def generate_model():
    config = ConfigParser()

    config_file = os.path.join(os.path.dirname('__file__'), 'config.ini')

    config.read(config_file)
    default = config['aws.data']
    aws_key = default['accessKey']
    aws_secret = default['secretAccessKey']

    bucket_name = 'pubg-dataset-files'
    object_key = 'final_train.csv'

    path = 's3://{}:{}@{}/{}'.format(aws_key, aws_secret, bucket_name, object_key)

    #df = pd.read_csv(smart_open(path))
    #df_train = pd.read_csv(smart_open(path))
    c_size = 2500000

    for gm_chunk in pd.read_csv(smart_open(path),skipinitialspace=True, error_bad_lines=False, index_col=False, dtype='unicode',
                                chunksize=c_size):
        #frames = [df_train, gm_chunk]
        df_train = gm_chunk
        # print(df_train.shape)
        break



    print(df_train.shape)
    print('read data')
    df_train = df_train.drop('Unnamed: 0', axis=1)
    X = df_train.drop('winPlacePerc', axis=1)
    y = df_train.winPlacePerc
    print('train test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print('start RF')
    rf = RandomForestRegressor(n_estimators=30, max_depth=7)
    rf.fit(X_train, y_train)
    #NeuralNetwork(X_train, y_train)
    print('NN done')
    #LinearRegree(X_train, y_train)
    print('LR done')
    print('done RF')
    #calc_error_metric('RandomForest', rf, X_train, y_train, X_test, y_test)
    pickle.dump(rf, open('random_forest.model', 'wb'))
    print('done pickle')

def LinearRegree(X_train, y_train):
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    pickle.dump(lm, open('linear_regression.model', 'wb'))


def NeuralNetwork(X_train, y_train):
    nn = MLPRegressor()
    nn.fit(X_train, y_train)
    pickle.dump(nn, open('neural_network.model', 'wb'))

def doPrediction():
    loaded_model = pickle.load(open('random_forest.model', 'rb'))
    trail = [244.80, 1, 0, 0, 0, 0, 0, 0, 0, 0, 60, 26]
    predict = loaded_model.predict([trail])
    print(predict)


def calc_error_metric(modelname, model, X_train_scale, y_train, X_test_scale, y_test):
    global error_metric
    y_train_predicted = model.predict(X_train_scale)
    y_test_predicted = model.predict(X_test_scale)

    # MAE, RMS, MAPE, R2
    error_metric = pd.DataFrame({'r2_train': [],
                                 'r2_test': [],
                                 'rms_train': [],
                                 'rms_test': [],
                                 'mae_train': [],
                                 'mae_test': []})

    rmse_dict = {}

    r2_train = r2_score(y_train, y_train_predicted)
    r2_test = r2_score(y_test, y_test_predicted)

    rms_train = sqrt(mean_squared_error(y_train, y_train_predicted))
    rms_test = sqrt(mean_squared_error(y_test, y_test_predicted))

    mae_train = mean_absolute_error(y_train, y_train_predicted)
    mae_test = mean_absolute_error(y_test, y_test_predicted)

    #     mape_train = np.mean(np.abs((y_train - y_train_predicted) / y_train)) * 100
    #     mape_test = np.mean(np.abs((y_test - y_test_predicted) / y_test)) * 100

    rmse_dict[modelname] = rms_test

    df_local = pd.DataFrame({'Model': [modelname],
                             'r2_train': [r2_train],
                             'r2_test': [r2_test],
                             'rms_train': [rms_train],
                             'rms_test': [rms_test],
                             'mae_train': [mae_train],
                             'mae_test': [mae_test]})

    error_metric = pd.concat([error_metric, df_local])
    return error_metric


def main():
    generate_model()
    doPrediction()


if __name__ == '__main__':
    main()

