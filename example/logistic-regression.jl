using MeLOne
using Plots
gr(size=(800,600))

function Logistic_regression()
  n = 400
  X = 2 * rand(n, 2)
  y = [X[i,1].^2 + X[i,2].^2 < 1.5 + randn() * 0.6 ? 0 : 1 for i = 1:n]

  model = LogisticRegression()
  fit!(model, X, y)

  xg = range(minimum(X[:,1]), maximum(X[:,1]), length=50)
  yg = range(minimum(X[:,2]), maximum(X[:,2]), length=50)
  pred(x,y) = predict(model, [x y])[1]

  heatmap(xg, yg, pred, c=cgrad([:pink, :lightblue]))
  scatter!(X[:,1], X[:,2], c=y, leg=false)
  y_pred = predict(model, X)
  title!("acc = $(accuracy_score(y, y_pred))")

  png("logistic-regression")

  X_ply = [X   X[:,1].^2   X[:,1] .* X[:,2]   X[:,2].^2]

  model = LogisticRegression()
  fit!(model, X_ply, y)

  xg = range(minimum(X[:,1]), maximum(X[:,1]), length=50)
  yg = range(minimum(X[:,2]), maximum(X[:,2]), length=50)
  pred_ply(x,y) = predict(model, [x y x^2 x*y y^2])[1]

  heatmap(xg, yg, pred_ply, c=cgrad([:pink, :lightblue]))
  scatter!(X[:,1], X[:,2], c=y, leg=false)
  y_pred = predict(model, X_ply)
  title!("acc = $(accuracy_score(y, y_pred))")

  png("logistic-regression-poly2")
end

Logistic_regression()
