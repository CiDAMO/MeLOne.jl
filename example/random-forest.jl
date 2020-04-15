using MeLOne
using Plots
using Random
gr(size=(800,600))

function randomforest()
  Random.seed!(0)
  n = 400
  X = 2 * rand(n, 2)
  y = [X[i,1].^2 + X[i,2].^2 < 1.5 + randn() * 0.0 ? 0 : 1 for i = 1:n]

  k = 1
  for max_depth = 6:10
    model = RandomForest(max_depth=max_depth, n_estimators=100)
    fit!(model, X, y)

    xg = range(minimum(X[:,1]), maximum(X[:,1]), length=200)
    yg = range(minimum(X[:,2]), maximum(X[:,2]), length=200)
    pred(x,y) = predict_proba(model, [x y])[1,1]
    #pred(x,y) = predict(model, [x y])[1]

    heatmap(xg, yg, pred, c=cgrad([:pink, :lightblue]))
    scatter!(X[:,1], X[:,2], c=y, leg=false)
    y_pred = predict(model, X)
    title!("RandomForest: acc = $(accuracy_score(y, y_pred)), max_depth=$max_depth")

    png("random-forest-$k")
    k += 1
  end
end

randomforest()
