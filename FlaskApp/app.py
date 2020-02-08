from flask import Flask, request, render_template
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

app = Flask(__name__)

# @ signifies a decorator - way to wrap a function and modify its behaviour
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/anamoly')
def anamoly():
    return render_template('anamoly.html')

@app.route('/notebook_prediction')
def notebook_prediction():
    return render_template('notebook_prediction.html')

@app.route('/submit',methods=['POST'])
def submit():
    name = request.form["name"]
    result()
    #return render_template('result.html',name)

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
     # model_name=request.form['Model_Name']

      predict=do_prediction(result)
      return render_template("result.html",result=predict)
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

    df_train = pd.read_csv(smart_open(path))
    df_train = df_train.drop('Unnamed: 0', axis=1)
    X = df_train.drop('winPlacePerc', axis=1)
    y = df_train.winPlacePerc
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    rf = RandomForestRegressor(n_estimators=30, max_depth=7)
    rf.fit(X_train, y_train)
    calc_error_metric('RandomForest', rf, X_train, y_train, X_test, y_test)
    pickle.dump(rf, open('random_forest.model', 'wb'))


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



def do_prediction(result):
    arr=[]
    print(type(result))

    for key, value in result.items():
        #print(type(value))
        a=value
        arr.append(a)
        #print(type(a))

    model =arr[0]
    print(model)
    arr = arr[1:]
    print(arr)
    ark=[]
    for a in arr:
        a=int(a)
        ark.append(a)
	filename=''
    if model=='Linear Regression':
        filename='linear_regression.model'
    elif model=='Random Forest':
        filename='random_forest.model'
    elif model=='Neural Network':
        filename='neural_network.model'
		
	urllib.request.urlretrieve("https://s3.amazonaws.com/finalprojectads/PUBG-Dataset-Files", filename=filename)
	loaded_model = pickle.load(open(filename, 'rb'))

    predict = loaded_model.predict([ark])
    print(predict)
    return predict




if __name__ == '__main__':
    app.run(host='0.0.0.0')

