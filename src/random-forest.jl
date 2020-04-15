export RandomForest

mutable struct RandomForest <: MeLOneModel
  _classes :: Vector
  trees :: Vector{DecisionTree}

  options :: Dict{Symbol,Any}
end

const random_forest_option_list = [(:min_impurity_decrease, 0.0), (:max_depth, 5), (:n_estimators, 100)]

function RandomForest(; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in random_forest_option_list)
  for k in keys(kwargs)
    if !(k in getfield.(random_forest_option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return RandomForest(Int[], DecisionTree[], options)
end

import Base.show
function show(io :: IO, model :: RandomForest)
  print(io, "🌲🌲🌲 RandomForest, $(model.options[:n_estimators]) trees")
end

function fit!(model :: RandomForest,
              X :: Matrix, y :: Vector)
  tree_options = Dict{Symbol, Any}(k => get(model.options, k, v) for (k,v) in decision_tree_option_list)
  tree_options[:splitter] = :random

  model._classes = unique(y)
  for n = 1:model.options[:n_estimators]
    tree = DecisionTree(;tree_options...)
    fit!(tree, X, y, classes=model._classes)
    push!(model.trees, tree)
  end

  return model
end

function predict(model :: RandomForest,
                 X :: Matrix)
  y_pred_mat = predict_proba(model, X)
  return model._classes[getindex.(argmax(y_pred_mat, dims=2)[:], 2)]
end

function predict_proba(model :: RandomForest,
                       X :: Matrix)
  n = size(X, 1)
  y_pred = zeros(n, length(model._classes))
  for tree in model.trees
    y_pred += predict_proba(tree, X)
  end
  y_pred /= length(model.trees)

  return y_pred
end
