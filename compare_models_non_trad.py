# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:29:31 2022

@author: ba42pizo
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score 
import sys
#sys.path.append('../features')
from build_features import prepare_final_files


def naive_bayes(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad):

    from sklearn.naive_bayes import GaussianNB

    modelnb = GaussianNB()
    modelnb.fit(x_train, y_train)
    score = modelnb.score(x_test, y_test)
    y_pred = modelnb.predict(x_test)
    f1 = f1_score(y_test, y_pred, average = 'binary')
    print("NB: acurracy: " + str(score) + "; f1 score: " + str(f1))
    # non trads
    score = modelnb.score(x_test_ntrad, y_test_ntrad)
    y_pred = modelnb.predict(x_test_ntrad)
    f1 = f1_score(y_test_ntrad, y_pred, average = 'binary')
    print("RF for non traditionals: acurracy: " + str(score) + "; f1 score: " + str(f1))
    #with open(os.path.abspath('../../models/modelnb_pickle'), 'wb') as file:
    #    pickle.dump(modelnb, file)

def random_forest(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad):
    
    from sklearn.ensemble import RandomForestClassifier
    
    modelrf = RandomForestClassifier(n_estimators = 1000)
    modelrf.fit(x_train, y_train)
    score = modelrf.score(x_test, y_test)
    y_pred = modelrf.predict(x_test)
    f1 = f1_score(y_test, y_pred, average = 'binary')
    print("RF: acurracy: " + str(score) + "; f1 score: " + str(f1))
    # non trads
    score = modelrf.score(x_test_ntrad, y_test_ntrad)
    y_pred = modelrf.predict(x_test_ntrad)
    f1 = f1_score(y_test_ntrad, y_pred, average = 'binary')
    print("RF for non traditionals: acurracy: " + str(score) + "; f1 score: " + str(f1))
    #with open(os.path.abspath('../../models/modelrf_pickle'), 'wb') as file:
    #    pickle.dump(modelrf, file)


def logistic_regression(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad):
    
    from sklearn.linear_model import LogisticRegression

    logReg = LogisticRegression()

    logReg.fit(x_train, y_train)
    score = logReg.score(x_test, y_test)
    y_pred = logReg.predict(x_test)
    f1 = f1_score(y_test, y_pred, average = 'binary')
    print("LR: acurracy: " + str(score) + "; f1 score: " + str(f1))
    # non trads
    score = logReg.score(x_test_ntrad, y_test_ntrad)
    y_pred = logReg.predict(x_test_ntrad)
    f1 = f1_score(y_test_ntrad, y_pred, average = 'binary')
    print("RF for non traditionals: acurracy: " + str(score) + "; f1 score: " + str(f1))
    #with open(os.path.abspath('../../models/modellr_pickle'), 'wb') as file:
    #    pickle.dump(logReg, file)

def support_vector_machine(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad):
    
    from sklearn.svm import SVC

    modelsvm = SVC(kernel = 'rbf')
    modelsvm.fit(x_train, y_train)
    score = modelsvm.score(x_test, y_test)
    y_pred = modelsvm.predict(x_test)
    f1 = f1_score(y_test, y_pred, average = 'binary')
    print("SVM: acurracy: " + str(score) + "; f1 score: " + str(f1))
    # non trads
    score = modelsvm.score(x_test_ntrad, y_test_ntrad)
    y_pred = modelsvm.predict(x_test_ntrad)
    f1 = f1_score(y_test_ntrad, y_pred, average = 'binary')
    print("RF for non traditionals: acurracy: " + str(score) + "; f1 score: " + str(f1))
    #with open(os.path.abspath('../../models/modelsvm_pickle'), 'wb') as file:
    #    pickle.dump(modelsvm, file)

def neural_net(x_train, y_train, x_test, y_test):
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    modelann = Sequential([
        Dense(units = 32, input_shape=(len(x_test.columns),), activation = 'sigmoid'),
        Dense(units = 64, activation = 'sigmoid'),
        Dense(units = 128, activation = 'sigmoid'),
        Dense(units = 256, activation = 'sigmoid'),
        Dense(units = 128, activation = 'sigmoid'),
        Dense(units = 64, activation = 'sigmoid'),
        Dense(units = 32, activation = 'sigmoid'),
        Dense(units = 1, activation = 'sigmoid')])

    modelann.compile(optimizer = Adam(learning_rate=0.001), loss= 'mean_squared_error', metrics=['accuracy'])
    modelann.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2, verbose = 0, callbacks=None, validation_data=None, shuffle=True)
    y_pred = modelann.predict(x_test)
    y_pred = np.where(y_pred >= 0.5, 1, 0)
    #y_test, y_pred = set(1, 2, 4), set(2, 8, 1)
    y = np.asarray(y_test) - y_pred[:,0]
    score = np.count_nonzero(y == 0)/(y_pred[:,0]).size
   #from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    #cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average = 'binary')
    print("ANN: acurracy: " + str(score) + "; f1 score: " + str(f1))
    #with open(os.path.abspath('C:../../models/modelann_pickle'), 'wb') as file:
    #    pickle.dump(modelann, file)

if __name__ == '__main__':
    #for master change path from bachelor to master 
    df_StudStart = pd.read_pickle(os.path.abspath('./interim/bachelor/df_studstart_without_prediction.pkl'))  
    df_Path = pd.read_pickle(os.path.abspath('./interim/bachelor/df_path.pkl'))
    df_demo = pd.read_pickle(os.path.abspath('./interim/bachelor/df_demo.pkl'))
    for semester in range(1,2):
        #for master add 'master in function
        Final = prepare_final_files(semester, df_StudStart, df_Path, df_demo, 'bachelor')[1]
        
        x_train, x_test, y_train, y_test = train_test_split(Final.drop(['MNR_Zweit', 'Startsemester', 'studiengang', 'final'], axis = 1), 
                                                            Final.final, test_size = 0.25, random_state = 0)
        Final2 = pd.merge(x_test, y_test, left_index=True, right_index=True)
        Final2 = Final2.loc[Final2['Beruflich qualifiziert'] == 1]
        x_train_ntrad, x_test_ntrad, y_train_ntrad, y_test_ntrad = train_test_split(Final2.drop(['MNR_Zweit', 'Startsemester', 'studiengang', 'final'], axis = 1), 
                                                            Final2.final, test_size = 0.25, random_state = 0)
        
        x_test_ntrad = Final2.drop(['final'], axis = 1)
        y_test_ntrad = Final2.final            
        logistic_regression(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad)
        random_forest(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad)
        naive_bayes(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad)
        support_vector_machine(x_train, y_train, x_test, y_test, x_test_ntrad, y_test_ntrad)
        neural_net(x_train, y_train, x_test, y_test)

import numpy as np
np.version
