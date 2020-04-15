using MeLOne
using Plots
gr(size=(600,400))

function linear_regression()
  n = 30
  x = sort(rand(n))
  y = exp.(x) + 0.1 * randn(n)
  model = LinearRegression()
  X = [ones(n)  x  x.^2]
  fit!(model, X, y)

  scatter(x, y, lab="training data")
  xg = range(0, 1, length=100)
  X = [ones(100)  xg  xg.^2]
  plot!(xg, predict(model, X), lab="model")
  title!("LinearRegression")

  png("linear-regression")
end

linear_regression()
