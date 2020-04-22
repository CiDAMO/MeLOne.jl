using LinearAlgebra, Test

@testset "KNNRegressor" begin
  n = 50
  X = rand(n, 2)
  y = X[:,1].^2 + X[:,1].^2 + randn(n) * 0.1
  model = KNNRegressor(n_neighbors = 1)
  fit!(model, X, y)
  @test predict(model, X) == y

  model = KNNRegressor(n_neighbors = 5, weights=:distance)
  fit!(model, X, y)
  @test r2_score(y, predict(model, X)) > 0.9
end
