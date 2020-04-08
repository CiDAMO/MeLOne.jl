using LinearAlgebra

export DecisionTree, fit!, predict, predict_proba

mutable struct Subset
  I :: Vector{Int}
  probs :: Vector{Float64} # Same size of _classes
end

mutable struct Node
  left :: Union{Node, Subset}
  right :: Union{Node, Subset}
  column :: Int
  threshold :: Real
end

mutable struct DecisionTree
  root :: Union{Node, Subset}
  _classes :: Vector

  options :: Dict{Symbol,Any}
end

const option_list = [(:min_impurity_decrease, 0.0), (:max_depth, Inf), (:splitter, :best)]

function DecisionTree(root, classes; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in option_list)
  for k in keys(kwargs)
    if !(k in getfield.(option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return DecisionTree(root, classes, options)
end

function DecisionTree(; kwargs...)
  s = Subset(Int[], Float64[])
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in option_list)
  for k in keys(kwargs)
    if !(k in getfield.(option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return DecisionTree(Node(s, s, 0, 0.0), Int[], options)
end

"""
    IG = 1 - ∑pₖ²
"""
function gini(classes, y)
  g = 1.0
  n = length(y)
  for k in classes
    g -= (count(y .== k) / n)^2
  end
  return g
end

"""
Escolher qual coluna e threshold para cortar o conjunto I em dois.
"""
function gini_split(I, X, y, classes, curr_depth;
                    max_depth=Inf,
                    splitter=:best,
                    min_impurity_decrease = 0.0)
  full_gini = gini(classes, y[I])
  p = size(X, 2)
  n = length(I)
  best_c = 0
  best_t = 0.0
  best_Iright = Int[]
  best_Ileft = Int[]
  best_gini = full_gini
  if splitter == :best
    for c = 1:p
      for i in I
        t = X[i,c]
        Iright = I[findall(X[I,c] .≥ t)]
        nright = length(Iright)
        (nright == 0 || nright == n) && continue
        Ileft  = setdiff(I, Iright)
        nleft  = n - nright
        gini_split = nright * gini(classes, y[Iright]) / n +
                     nleft  * gini(classes, y[Ileft])  / n
        if gini_split ≤ best_gini
          best_gini = gini_split
          best_c = c
          best_t = t
          best_Iright = copy(Iright)
          best_Ileft  = copy(Ileft)
        end
      end
    end
  elseif splitter == :random
    best_c = rand(1:p)
    best_t = rand(sort(X[I,best_c])[2:end])
    best_Iright = I[findall(X[I,best_c] .≥ best_t)]
    best_Ileft = setdiff(I, best_Iright)
    best_gini = length(best_Iright) * gini(classes, y[best_Iright]) / n +
                length(best_Ileft)  * gini(classes, y[best_Ileft])  / n
  else
    @error "Unknown value for parameter splitter: $splitter. Possible values are :best and :random"
  end
  Δgini = (full_gini - best_gini) * n / length(y)
  stopnow = (Δgini < min_impurity_decrease) ||
            (curr_depth == max_depth - 1)
  left = if length(unique(y[best_Ileft])) == 1
    # In the future, allow early stop
    probs = zeros(length(classes))
    c = y[best_Ileft[1]]
    probs[findfirst(c .== classes)] = 1.0
    Subset(best_Ileft, probs)
  elseif stopnow
    # In the future, allow early stop
    probs = [sum(y[best_Ileft] .== c) for c in classes] / length(best_Ileft)
    Subset(best_Ileft, probs)
  else
    gini_split(best_Ileft, X, y, classes, curr_depth+1;
               max_depth=max_depth,
               splitter=splitter,
               min_impurity_decrease=min_impurity_decrease
              )
  end
  right = if length(unique(y[best_Iright])) == 1
    # In the future, allow early stop
    probs = zeros(length(classes))
    c = y[best_Iright[1]]
    probs[findfirst(c .== classes)] = 1.0
    Subset(best_Iright, probs)
  elseif stopnow
    # In the future, allow early stop
    probs = [sum(y[best_Iright] .== c) for c in classes] / length(best_Iright)
    Subset(best_Iright, probs)
  else
    gini_split(best_Iright, X, y, classes, curr_depth+1;
               max_depth=max_depth,
               splitter=splitter,
               min_impurity_decrease=min_impurity_decrease
              )
  end
  return Node(left, right, best_c, best_t)
end

function fit!(model :: DecisionTree,
              X :: Matrix, y :: Vector)
  n = size(X, 1)
  I = 1:n
  model._classes = unique(y)
  model.root = gini_split(I, X, y, model._classes, 0;
                          max_depth=model.options[:max_depth],
                          min_impurity_decrease=model.options[:min_impurity_decrease],
                          splitter=model.options[:splitter]
                         )

  return model
end

function predict(model :: DecisionTree,
                 X :: Matrix)

  n = size(X, 1)
  y_pred = Vector{eltype(model._classes)}(undef, n)
  for i = 1:n
    node = model.root
    while node isa Node
      col, thr = node.column, node.threshold
      node = X[i,col] ≥ thr ? node.right : node.left
    end
    # Now node is a Subset
    probs = node.probs
    y_pred[i] = model._classes[argmax(probs)]
  end

  return y_pred
end

function predict_proba(model :: DecisionTree,
                 X :: Matrix)

  n = size(X, 1)
  y_pred = zeros(n, length(model._classes))
  for i = 1:n
    node = model.root
    while node isa Node
      col, thr = node.column, node.threshold
      node = X[i,col] ≥ thr ? node.right : node.left
    end
    # Now node is a Subset
    y_pred[i,:] .= node.probs
  end

  return y_pred
end
