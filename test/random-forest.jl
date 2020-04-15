using LinearAlgebra, Test

@testset "RandomForest" begin
  n = 400
  X = 2 * rand(n, 2)
  y = [X[i,1]^2 + X[i,2]^2 < 1.5 ? :red : :blue for i = 1:n]
  model = RandomForest(max_depth=10)
  fit!(model, X, y)
  @test accuracy_score(y, predict(model, X)) > 0.99 # Overfitting with luck

  X = randn(n, 2)
  y = [(X[i,1] > 0 ? 0 : 1) + (X[i,2] > 0 ? 1 : 3) for i = 1:n]
  model = RandomForest(max_depth=10)
  fit!(model, X, y)
  @test accuracy_score(y, predict(model, X)) > 0.99 # Overfitting with luck

  # Creating fake "Random" forest with best splitter decision tree
  X = [i + j for i = -1:0.01:1, j = 1:10]
  n = size(X, 1)
  y = rand(0:1, n)
  model = RandomForest()
  tree = DecisionTree() # exhaustive ⇒ overfitting
  fit!(tree, X, y)
  model._classes = tree._classes
  model.trees = [tree]
  model.options[:n_estimators] = 1
  @test predict(model, X) == y
end
