import pathlib
import sys
import yaml
import joblib


import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(train_features,target, n_estimators,max_depth,seed):

    model = RandomForestClassifier
    (n_estimators=n_estimators, max_depth=max_depth,random_state=seed)
    model.fit(train_features,target)
    
  