# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 17:51:21 2021

@author: Pierre
"""

import optuna
from sklearn.model_selection import cross_val_score
from pipeline_utils import make_pipeline
from utils import load_data

def objective(trial):
    X_train_resampled, y_train_resampled, _ ,_ = load_data()
    
    params = {
        'learning_rate' : trial.suggest_loguniform("learning_rate", 1e-4, 10),
        'n_estimators' : trial.suggest_int("n_estimators", 1, 400),
        'max_depth' : trial.suggest_int("max_depth", 2, 10),
        'min_split_gain' : trial.suggest_uniform("min_split_gain", 0, 1),
        'device' : 'gpu'
        }
    
    pipeline = make_pipeline(params)
    
    score = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv = 4).mean()

    return score

if __name__ == '__main__':
    study = optuna.load_study(study_name = 'ml_speedrun_lightgbm', storage = 'sqlite:///trials.db')
    study.optimize(objective, n_trials=100)