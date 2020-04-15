# MeLOne

**M**achin**e** **L**earning **One** - A package with simple implementation of Machine Learning models, made live on [twitch](https://twitch.tv/abelsiqueira).

[![Build Status](https://travis-ci.org/CiDAMO/MeLOne.jl.svg?branch=master)](https://travis-ci.org/CiDAMO/MeLOne.jl)

## Methods

We have implemented a simple version of the following methods:

*Regression*:
- Linear Regression

*Classification*:
- DecisionTree
- KNN
- Logistic Regression
- RandomForest

## API

Our objective is that all our methods implement the following API:

- `model = Model()`
- `fit!(model, X, y)`
- `y_pred = predict(model, X)`

In addition, some models will implement

- `pred_matrix = predict_proba(model, X)`
