# MeLOne

**M**achin**e** **L**earning **One** - A package with simple implementation of Machine Learning models, made live on [twitch](https://twitch.tv/abelsiqueira).

[![Build Status](https://travis-ci.org/CiDAMO/MeLOne.jl.svg?branch=master)](https://travis-ci.org/CiDAMO/MeLOne.jl)
[![Coverage Status](https://coveralls.io/repos/github/CiDAMO/MeLOne.jl/badge.svg?branch=master)](https://coveralls.io/github/CiDAMO/MeLOne.jl?branch=master)

## Methods

We have implemented a simple version of the following methods:

*Regression*:
- DecisionTreeRegressor
- LinearRegression
- KNNRegressor

*Classification*:
- DecisionTreeClassifier
- KNNClassifier
- LogisticRegression
- RandomForest

## API

Our objective is that all our methods implement the following API:

- `model = Model()`
- `fit!(model, X, y)`
- `y_pred = predict(model, X)`

In addition, some models will implement

- `pred_matrix = predict_proba(model, X)`

## Examples

<img src="example/decision-tree-classifier-8.png" height="400">
<img src="example/decision-tree-regressor-3.png" height="400">
<img src="example/knn-classifier.png" height="400">
<img src="example/knn-regressor-4.png" height="400">
<img src="example/linear-regression.png" height="400">
<img src="example/logistic-regression.png" height="400">
<img src="example/random-forest-5.png" height="400">
