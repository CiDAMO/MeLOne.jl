using LinearAlgebra

export DecisionTree, fit!, predict

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
end

function DecisionTree()
  s = Subset(Int[], Float64[])
  return DecisionTree(Node(s, s, 0, 0.0), Int[])
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
function gini_split(I, X, y, classes)
  full_gini = gini(classes, y[I])
  p = size(X, 2)
  n = length(I)
  best_c = 0
  best_t = 0.0
  best_Iright = Int[]
  best_Ileft = Int[]
  best_gini = full_gini
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
  left = if length(unique(y[best_Ileft])) == 1
    # In the future, allow early stop
    probs = zeros(length(classes))
    c = y[best_Ileft[1]]
    probs[findfirst(c .== classes)] = 1.0
    Subset(best_Ileft, probs)
  else
    gini_split(best_Ileft, X, y, classes)
  end
  right = if length(unique(y[best_Iright])) == 1
    # In the future, allow early stop
    probs = zeros(length(classes))
    c = y[best_Iright[1]]
    probs[findfirst(c .== classes)] = 1.0
    Subset(best_Iright, probs)
  else
    gini_split(best_Iright, X, y, classes)
  end
  return Node(left, right, best_c, best_t)
end

function fit!(model :: DecisionTree,
              X :: Matrix, y :: Vector)
  n = size(X, 1)
  I = 1:n
  model._classes = unique(y)
  model.root = gini_split(I, X, y, model._classes)

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
