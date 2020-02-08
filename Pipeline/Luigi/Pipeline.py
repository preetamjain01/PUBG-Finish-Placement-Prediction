import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import *
from math import sqrt
import operator
import pickle
import luigi


class clean_data(luigi.Task):


    def run(self):
        train = pd.read_csv('C:\SGN_ADS_final_project\PUBG_DataSet\data_train.csv')
        train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')
        train['killsNorm'] = train['kills'] * ((100 - train['playersJoined']) / 100 + 1)
        train['damageDealtNorm'] = train['damageDealt'] * ((100 - train['playersJoined']) / 100 + 1)

        train['healsAndBoosts'] = train['heals'] + train['boosts']
        train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
        train['boostsPerWalkDistance'] = train['boosts'] / (train[
                                                                'walkDistance'] + 1)  # The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
        train['boostsPerWalkDistance'].fillna(0, inplace=True)
        train['healsPerWalkDistance'] = train['heals'] / (train[
                                                              'walkDistance'] + 1)  # The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
        train['healsPerWalkDistance'].fillna(0, inplace=True)
        train['healsAndBoostsPerWalkDistance'] = train['healsAndBoosts'] / (
                    train['walkDistance'] + 1)  # The +1 is to avoid infinity.
        train['healsAndBoostsPerWalkDistance'].fillna(0, inplace=True)
        train.drop(2744604, inplace=True)
        train.drop(train[train['kills'] > 30].index, inplace=True)
        train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
        train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
        train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
        train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
        train.drop(train[train['heals'] >= 40].index, inplace=True)
        train['killsPerWalkDistance'] = train['kills'] / (train[
                                                              'walkDistance'] + 1)  # The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
        train['killsPerWalkDistance'].fillna(0, inplace=True)
        train['team'] = [1 if i > 50 else 2 if (i > 25 & i <= 50) else 4 for i in train['numGroups']]

        cols = ['totalDistance', 'weaponsAcquired', 'healsAndBoosts', 'longestKill', 'killsNorm', 'assists', 'DBNOs',
                'headshotKills',
                'revives', 'vehicleDestroys', 'winPlacePerc', 'killPlace', 'numGroups']
        final_train = train[cols]
        final_train[['totalDistance', 'healsAndBoosts', 'killsNorm']] = final_train[['totalDistance',
                                                                                     'healsAndBoosts',
                                                                                     'killsNorm']].astype('float64')

         # Making a CSV of this cleaned data
        final_train.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("final_train1.csv")

class run_models(luigi.Task):
    def requires(self):
        yield clean_data()

    def run(self):
        X = pd.read_csv(clean_data().output().path)
        # Target label is 'ClassNumber'
        y = pd.DataFrame(X['winPlacePerc'])

        # Removing label from dataframe
        X = X.drop(['winPlacePerc'], axis=1)

        # Dividing data to training and testing sets with test size of 20%
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Opening a file for saving model accuracy
        file = open('Model_Accuracy.txt', mode='w')
        file.write('Model' + "," + 'Train Accuracy' + "," + 'Test Accuracy')
        file.write("\n")
        # Opening a file for saving error metrics
        file1 = open('Model_Error_Metrics.txt', mode='w')

        error_metric = pd.DataFrame({'r2_train': [],
                                     'r2_test': [],
                                     'rms_train': [],
                                     'rms_test': [],
                                     'mae_train': [],
                                     'mae_test': []})

        rmse_dict = {}

        def calc_error_metric(modelname, model, X_train_scale, y_train, X_test_scale, y_test):
            global error_metric
            error_metric = pd.DataFrame({'r2_train': [],
                                         'r2_test': [],
                                         'rms_train': [],
                                         'rms_test': [],
                                         'mae_train': [],
                                         'mae_test': []})

            rmse_dict = {}
            y_train_predicted = model.predict(X_train_scale)
            y_test_predicted = model.predict(X_test_scale)

            # MAE, RMS, MAPE, R2

            r2_train = r2_score(y_train, y_train_predicted)
            r2_test = r2_score(y_test, y_test_predicted)

            rms_train = sqrt(mean_squared_error(y_train, y_train_predicted))
            rms_test = sqrt(mean_squared_error(y_test, y_test_predicted))

            mae_train = mean_absolute_error(y_train, y_train_predicted)
            mae_test = mean_absolute_error(y_test, y_test_predicted)

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

        # creating a function which accepts parameters of training and testing data sets
        def model_Implementation(X_train, y_train, X_test, y_test):

            models = [
                RandomForestRegressor(max_depth=8, random_state=0),
                LinearRegression(),
                MLPRegressor(hidden_layer_sizes=(30, 50, 50))
            ]

            TestModels = []
            i = 0
            for model in models:
                # get model name

                # fit model on training dataset

                i = i + 1
                model.fit(X_train, y_train)

                if i == 1:
                    filename = 'RandomForestRegressor.sav'
                    TestModels.append(filename)
                elif i == 2:
                    filename = 'LinearRegression.sav'
                    TestModels.append(filename)
                elif i == 3:
                    filename = 'MLPRegressor.sav'
                    TestModels.append(filename)
                # elif i == 4:
                #     filename = 'MLPClassifier.sav'
                #     TestModels.append(filename)

                pickle.dump(model, open(filename, 'wb'))

                file1.write(filename)
                file1.write("\n")

                predictions = model.predict(X_test)
                predictions_trn = model.predict(X_train)

                accuracy_train = model.score(X_train, y_train)
                # print("Accuracy of the training is :", accuracy_train)

                cm = calc_error_metric(filename,model,X_train, y_train, X_test, y_test)
                # print(cm)
                file1.write(" Error Metrics:")
                file1.write("\n")
                file1.write(str(cm))
                file1.write("\n")

                accuracy_test = model.score(X_test, y_test)
                # print("Accuracy of the testing is:", accuracy_test)
                data = filename + "," + str(accuracy_train) + "," + str(accuracy_test)
                file.write(data)
                file.write("\n")
            file.close()
            file1.close()
            return TestModels

        # Running the model
        TestModels = model_Implementation(X_train, y_train, X_test, y_test)
        TestModels.append('Model_Accuracy.txt')
        TestModels.append('Model_Error_Metrics.txt')
        model_file_names = pd.DataFrame(TestModels)
        # Writing
        model_file_names.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("model_file_names.csv")
if __name__=='__main__':
    luigi.build([clean_data(), run_models()], local_scheduler=True)

