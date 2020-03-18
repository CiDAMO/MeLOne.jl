using LinearAlgebra

export KNNClassifier, fit!, predict

mutable struct KNNClassifier
  X :: Matrix
  y :: Vector

  _classes :: Vector

  n_neighbors :: Int
end

function KNNClassifier(; n_neighbors = 5)
  return KNNClassifier(zeros(2,0), [], [], n_neighbors)
end

function fit!(model :: KNNClassifier,
              X :: Matrix, y :: Vector)
  model.X = X
  model.y = y
  model._classes = unique(y)
  return model
end

function predict(model :: KNNClassifier,
                 X :: Matrix)
  k = model.n_neighbors
  n, p = size(X)
  n_tr = length(model.y)

  counters = zeros(Int, length(model._classes))
  y_pred = Vector{eltype(model._classes)}(undef, n)
  for i = 1:n
    fill!(counters, 0)
    D = [norm(X[i,:] - model.X[j,:]) for j = 1:n_tr]
    I = sortperm(D)
    for j = I[1:k]
      counters[findfirst(model.y[j] .== model._classes)] += 1
    end
    y_pred[i] = model._classes[argmax(counters)]
  end

  return y_pred
end
