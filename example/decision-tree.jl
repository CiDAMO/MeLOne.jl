using MeLOne
using Plots
using Random
gr(size=(800,600))

function decisiontree()
  Random.seed!(0)
  n = 400
  X = 2 * rand(n, 2)
  y = [X[i,1].^2 + X[i,2].^2 < 1.5 + randn() * 0.6 ? 0 : 1 for i = 1:n]

  k = 1
  for max_depth = 1:4, spl in [:best, :random]
    model = DecisionTree(max_depth=max_depth, splitter=spl)
    fit!(model, X, y)

    xg = range(minimum(X[:,1]), maximum(X[:,1]), length=200)
    yg = range(minimum(X[:,2]), maximum(X[:,2]), length=200)
    pred(x,y) = predict_proba(model, [x y])[1,1]

    heatmap(xg, yg, pred, c=cgrad([:pink, :lightblue]))
    scatter!(X[:,1], X[:,2], c=y, leg=false)
    y_pred = predict(model, X)
    title!("DecisionTree: acc = $(accuracy_score(y, y_pred)), max_depth=$max_depth, splitter=$spl")

    png("decision-tree-$k")
    k += 1
  end
end

decisiontree()
