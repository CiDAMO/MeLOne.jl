using MeLOne
using Plots
gr(size=(800,600))

function knn_regressor()
  n = 30
  x = collect(range(-1, 2, length=n))
  y = x.^2 + 2x .+ 3 + randn(n) * 0.2
  X = reshape(x, n, 1)

  c = 1
  for k = 1:2:7, w in [:uniform, :distance]
    model = KNNRegressor(n_neighbors=k, weights=w)
    fit!(model, X, y)
    y_pred = predict(model, X)

    xg = collect(range(extrema(x)..., length=100))
    yg = predict(model, reshape(xg, 100, 1))

    scatter(X, y, c=:blue, leg=false)
    plot!(xg, yg, c=:red, lw=2)

    r2 = round(r2_score(y, y_pred), digits=2)
    title!("KNNRegressor: n_neighbors=$k, weights=$w, r2 = $r2")

    png("knn-regressor-$c")
    c += 1
  end
end

knn_regressor()
