export DecisionTreeRegressor

mutable struct NodeReg
  left :: Union{NodeReg, Vector{Int}}
  right :: Union{NodeReg, Vector{Int}}
  column :: Int
  threshold :: Real
end

mutable struct DecisionTreeRegressor <: MeLOneModel
  root :: Union{NodeReg, Vector{Int}}
  y :: Vector

  options :: Dict{Symbol,Any}
end

import Base.show
function show(io :: IO, tree :: DecisionTreeRegressor)
  print(io, "🌲 DecisionTreeRegressor")
end

const decision_tree_regressor_option_list = [(:min_impurity_decrease, 0.0), (:max_depth, Inf), (:splitter, :best)]

function DecisionTreeRegressor(root, y; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in decision_tree_regressor_option_list)
  for k in keys(kwargs)
    if !(k in getfield.(decision_tree_regressor_option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return DecisionTreeRegressor(root, y, options)
end

function DecisionTreeRegressor(; kwargs...)
  options = Dict{Symbol,Any}(k => get(kwargs, k, v) for (k,v) in decision_tree_regressor_option_list)
  for k in keys(kwargs)
    if !(k in getfield.(decision_tree_regressor_option_list, 1))
      @warn "Keyword argument $k ignored"
    end
  end

  return DecisionTreeRegressor(NodeReg(Int[], Int[], 0, 0.0), Float64[], options)
end

"""
    regressor_impurity(y)

∑ᵢ(yᵢ - ȳ)² / n
"""
function regressor_impurity(y)
  sum((y .- mean(y)).^2) / length(y)
end

"""
Escolher qual coluna e threshold para cortar o conjunto I em dois.
"""
function regressor_split(I, X, y, curr_depth;
                         max_depth=Inf,
                         splitter=:best,
                         min_impurity_decrease = 0.0)
  full_impurity = regressor_impurity(y[I])
  p = size(X, 2)
  n = length(I)
  best_c = 0
  best_t = 0.0
  best_Iright = Int[]
  best_Ileft = Int[]
  best_impurity = full_impurity
  if splitter == :best
    for c = 1:p
      for i in I
        t = X[i,c]
        Iright = I[findall(X[I,c] .≥ t)]
        nright = length(Iright)
        (nright == 0 || nright == n) && continue
        Ileft  = setdiff(I, Iright)
        nleft  = n - nright
        impurity_split = nright * regressor_impurity(y[Iright]) / n +
                         nleft  * regressor_impurity(y[Ileft])  / n
        if impurity_split ≤ best_impurity
          best_impurity = impurity_split
          best_c = c
          best_t = t
          best_Iright = copy(Iright)
          best_Ileft  = copy(Ileft)
        end
      end
    end
  elseif splitter == :random
    c_opts = collect(1:p)
    best_c = rand(c_opts)
    t_opts = sort(unique(X[I,best_c]))
    while length(t_opts) == 1
      setdiff!(c_opts, best_c)
      if length(c_opts) == 0
        error("Unexpected error: no options for column split. Please open an issue on MeLOne.jl")
      end
      best_c = rand(c_opts)
      t_opts = sort(unique(X[I,best_c]))
    end
    best_t = rand(t_opts[2:end])
    best_Iright = I[findall(X[I,best_c] .≥ best_t)]
    best_Ileft = setdiff(I, best_Iright)
    best_impurity = length(best_Iright) * regressor_impurity(y[best_Iright]) / n +
                    length(best_Ileft)  * regressor_impurity(y[best_Ileft])  / n
  else
    @error "Unknown value for parameter splitter: $splitter. Possible values are :best and :random"
  end

  has_diff_rows = [false, false]
  for (c,II) in enumerate([best_Ileft, best_Iright])
    for i = II, j = II
      i ≤ j && continue
      if X[i,:] != X[j,:]
        has_diff_rows[c] = true
        break
      end
    end
  end
  Δimpurity = (full_impurity - best_impurity) * n / length(y)
  stopnow = (Δimpurity < min_impurity_decrease) ||
            (curr_depth == max_depth - 1)

  left = if length(unique(y[best_Ileft])) == 1 || stopnow || !has_diff_rows[1]
    best_Ileft
  else
    regressor_split(best_Ileft, X, y, curr_depth+1;
                    max_depth=max_depth,
                    splitter=splitter,
                    min_impurity_decrease=min_impurity_decrease
                   )
  end
  right = if length(unique(y[best_Iright])) == 1 || stopnow || !has_diff_rows[2]
    best_Iright
  else
    regressor_split(best_Iright, X, y, curr_depth+1;
                    max_depth=max_depth,
                    splitter=splitter,
                    min_impurity_decrease=min_impurity_decrease
                   )
  end
  return NodeReg(left, right, best_c, best_t)
end

function fit!(model :: DecisionTreeRegressor,
              X :: Matrix, y :: Vector)
  n = size(X, 1)
  I = 1:n
  model.root = regressor_split(I, X, y, 0;
                               max_depth=model.options[:max_depth],
                               min_impurity_decrease=model.options[:min_impurity_decrease],
                               splitter=model.options[:splitter]
                              )
  model.y = y

  return model
end

function predict(model :: DecisionTreeRegressor,
                 X :: Matrix)

  n = size(X, 1)
  y_pred = zeros(eltype(model.y), n)
  for i = 1:n
    node = model.root
    while node isa NodeReg
      col, thr = node.column, node.threshold
      node = X[i,col] ≥ thr ? node.right : node.left
    end
    # Now node is a Subset
    y_pred[i] = mean(model.y[node])
  end

  return y_pred
end
