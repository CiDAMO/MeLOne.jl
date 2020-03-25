export LogisticRegression, fit!, predict, predict_proba

using JSOSolvers, LinearAlgebra, Logging, NLPModels

mutable struct LogisticRegression
  β :: Vector
  threshold :: Float64
  λ :: Float64
end

function LogisticRegression(; threshold = 0.5, λ = 1e-1)
  return LogisticRegression([], threshold, λ)
end

function fit!(model :: LogisticRegression,
              X :: Matrix, y :: Vector)
  σ(t) = 1 / (1 + exp(-t))
  h(β, x) = σ(β[1] + dot(β[2:end], x))
  n, p = size(X)
  ℓ(β) = -sum(y[i] * log(h(β, X[i,:])) + (1 - y[i]) * log(1 - h(β, X[i,:])) for i = 1:n) + model.λ * dot(β, β) / 2
  nlp = ADNLPModel(ℓ, ones(p + 1))
  output = with_logger(NullLogger()) do
    JSOSolvers.trunk(nlp)
  end
  model.β = output.solution
  return model
end

function predict(model :: LogisticRegression,
                 X :: Matrix; prob = false)
  σ(t) = 1 / (1 + exp(-t))
  h(x) = σ(model.β[1] + dot(model.β[2:end], x))
  n = size(X, 1)
  if prob
    return [h(X[i,:]) for i = 1:n]
  else
    return [h(X[i,:]) > model.threshold ? 1 : 0 for i = 1:n]
  end
end

predict_proba(model :: LogisticRegression, X :: Matrix) = predict(model, X, prob=true)
