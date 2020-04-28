export KNNRegressor

mutable struct KNNRegressor <: MeLOneModel
  X :: Matrix
  y :: Vector

  options :: Dict{Symbol, Any}
end

const knn_regressor_option_list = [(:n_neighbors, 5), (:weights, :uniform)]

function KNNRegressor(; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in knn_regressor_option_list)
  for k in keys(kwargs)
    if !(k in getfield.(knn_regressor_option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return KNNRegressor(zeros(0,0), [], options)
end

function fit!(model :: KNNRegressor,
              X :: Matrix, y :: Vector)
  model.X = X
  model.y = y
  return model
end

function predict(model :: KNNRegressor,
                 X :: Matrix)
  k = model.options[:n_neighbors]
  n, p = size(X)
  n_tr = length(model.y)

  w_opt = model.options[:weights]

  y_pred = zeros(eltype(model.y), n)
  for i = 1:n
    D = [norm(X[i,:] - model.X[j,:]) for j = 1:n_tr]
    I = sortperm(D)
    ws = 0.0
    for j = 1:k
      ij = I[j]
      if w_opt == :uniform
        y_pred[i] += model.y[ij]
        ws += 1
      elseif w_opt == :distance
        w = 1 / max(D[ij], 1e-16)
        y_pred[i] += model.y[ij] * w
        ws += w
      elseif w_opt isa Vector
        y_pred[i] += model.y[ij] * w_opt[j]
        ws += w_opt[j]
      else
        y_pred[i] += model.y[ij] * w_opt(D[ij])
        ws += w_opt(D[ij])
      end
    end
    y_pred[i] /= ws
  end

  return y_pred
end
