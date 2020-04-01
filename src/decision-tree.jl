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
  return DecisionTree(Node(s, s, 0, 0.0))
end

function fit!(model :: DecisionTree,
              X :: Matrix, y :: Vector)

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
