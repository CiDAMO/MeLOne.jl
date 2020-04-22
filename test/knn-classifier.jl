using LinearAlgebra, Test

@testset "KNNClassifier" begin
  n = 50
  X = rand(n, 2)
  y = [X[i,1] + X[i,2] < 1 ? 0 : 1 for i = 1:n]
  model = KNNClassifier(n_neighbors = 1)
  fit!(model, X, y)

  @test predict(model, X) == y

  X = [1 1
       1 2
       2 1
       2 2
       3 2
       2 3
       3 3]
  y = [:blue, :blue, :blue, :red, :red, :red, :red]
  model = KNNClassifier(n_neighbors = 3)
  fit!(model, X, y)
  X_te = [0.5 0.5;
          1.5 1.5;
          2.5 2.5;
          3.5 3.5]
  @test predict(model, X_te) == [:blue, :blue, :red, :red]
  model.n_neighbors = 7
  X_te = rand(100, 2)
  @test all(predict(model, X_te) .== :red)
end
