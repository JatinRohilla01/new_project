import pathlib
import sys
import yaml
import joblib


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(train_features,target, n_estimators,max_depth,seed):

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=seed)
    model.fit(train_features,target)
    return model

def save_model(model, output_path):
    #saving the model to specified output path
    #Joblib is a Python library that provides a set of utilities for running computationally intensive tasks in parallel.To save a Python object using Joblib, you can use the joblib.dump() function. This function takes two arguments: the object to be saved and the filename to which the object should be saved.
    joblib.dump(model,output_path + '/model.joblib')

def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'

    params = yaml.safe_load(open(output_path))["train_model"]

    input_file = sys.argv[1]
    data_path= home_dir.as_posix() + input_file
    output_path = home_dir.as_posix()+ '/models'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok =True)

    TARGET = 'Class'
    train_features = pd.read_csv(data_path + '/train.csv')
    x = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    trained_model= train_model(x, y, params['n_estimators'], params['max_depth'], params['seed'])
