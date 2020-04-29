# All models should implement this API
export MeLOneModel, fit!, predict, predict_proba

abstract type MeLOneModel end

"""
    fit!(model, X, y)

Fit the `model` with matrix `X` and target `y`.
"""
function fit! end

"""
    y_pred = predict(model, X)

Predict the target of matrix `X` with `model`. `y_pred` is a Vector.
"""
function predict end

"""
    prob_mat = predict_proba(model, X)

Predict the probability of each row of matrix `X` being of each class
in `model._classes`. `prob_mat` is a matrix with dimensions `size(X, 1)`
by `length(model._classes)`. Each row of `prob_mat` that sum 1.
"""
function predict_proba end
