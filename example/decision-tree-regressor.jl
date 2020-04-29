using MeLOne
using Plots
gr(size=(800,600))

function decision_tree_regressor()
  n = 30
  x = collect(range(-1, 2, length=n))
  y = x.^2 + 2x .+ 3 + randn(n) * 0.2
  X = reshape(x, n, 1)

  c = 1
  for s in [:best, :random], d = 1:4
    model = DecisionTreeRegressor(max_depth=d, splitter=s)
    fit!(model, X, y)
    y_pred = predict(model, X)

    xg = collect(range(extrema(x)..., length=100))
    yg = predict(model, reshape(xg, 100, 1))

    scatter(X, y, c=:blue, leg=false)
    plot!(xg, yg, c=:red, lw=2)

    r2 = round(r2_score(y, y_pred), digits=2)
    title!("DecisionTreeRegressor: max_depth=$d, splitter=$s, r2 = $r2")

    png("decision-tree-regressor-$c")
    c += 1
  end
end

decision_tree_regressor()
