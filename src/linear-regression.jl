export LinearRegression, fit!, predict

using Krylov

mutable struct LinearRegression
  β :: Vector
end

function LinearRegression()
  return LinearRegression([])
end

function fit!(model :: LinearRegression,
              X :: Matrix, y :: Vector)
  model.β, krylov_stats = Krylov.cgls(X, y)
  return model
end

function predict(model :: LinearRegression,
                 X :: Matrix)
  return X * model.β
end
