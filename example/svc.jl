using MeLOne
using Plots, Random
gr(size=(800,600))

function svc()
  Random.seed!(0)
  n = 100
  X = 2 * rand(n, 2)
  y = [X[i,1].^2 + X[i,2].^2 < 1.5 + randn() * 0.6 ? -1 : 1 for i = 1:n]

  for kernel in [:rbf, :linear, :poly, :sigmoid]
    model = SVC(kernel=kernel, C=1e6, gamma=1.0, coef0=1.0, degree=4)
    fit!(model, X, y)

    xg = range(minimum(X[:,1]), maximum(X[:,1]), length=50)
    yg = range(minimum(X[:,2]), maximum(X[:,2]), length=50)
    pred(x,y) = predict(model, [x y])[1]

    heatmap(xg, yg, pred, c=cgrad([:pink, :lightblue]))
    I = findall(y .== -1)
    scatter!(X[I,1], X[I,2], c=:red, leg=false)
    I = findall(y .== 1)
    scatter!(X[I,1], X[I,2], c=:blue, leg=false)
    y_pred = predict(model, X)
    title!("SVC - kernel $kernel: acc = $(accuracy_score(y, y_pred))")
    png("svc-$kernel")
  end
end

svc()
