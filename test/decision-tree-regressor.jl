using LinearAlgebra, Test

@testset "DecissionTreeRegressor" begin
  g = range(0.0, 1.0, length=10)
  n = length(g)
  X = [repeat(g, inner=n) repeat(g, outer=n)]
  n = n^2
  y = X[:,1].^2 + X[:,2].^2 + randn(n) * 0.1
  model = DecissionTreeRegressor()
  fit!(model, X, y)
  @test predict(model, X) == y

  model = DecissionTreeRegressor()
  fit!(model, X, y)
  @test r2_score(y, predict(model, X)) > 0.9
end
