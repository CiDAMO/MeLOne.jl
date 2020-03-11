using LinearAlgebra, Test

@testset "Linear Regression" begin
  x = sort(rand(10))
  y = 2x .+ 3
  model = LinearRegression()
  X = [ones(10) x]
  fit!(model, X, y)
  @test model.β ≈ [3.0; 2.0]

  @test predict(model, X) ≈ y

  x = sort(rand(100))
  y = 2x .+ 3 + randn(100)
  model = LinearRegression()
  X = [ones(100) x]
  fit!(model, X, y)
  @test isapprox(norm(X' * (X * model.β - y)), 0.0, atol=1e-12 * cond(X))
end
