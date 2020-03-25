using LinearAlgebra, Test

@testset "Logistic Regression" begin
  X = reshape([0.0; 0.4; 0.6; 1.0], 4, 1)
  y = [0; 1; 0; 1]
  model = LogisticRegression()
  fit!(model, X, y)
  @test predict(model, X) == [0; 0; 1; 1]
  y_pred = predict(model, X)
  @test accuracy_score(y, y_pred) == 0.5

  n = 50
  X = [randn(div(n, 2), 2) .- 3;
       randn(div(n, 2), 2) .+ 3]
  y = [X[i,1] + X[i,2] < 0 ? 0 : 1 for i = 1:n]
  model = LogisticRegression(λ = 1e-2)
  fit!(model, X, y)
  y_pred = predict(model, X)
  @test accuracy_score(y, y_pred) == 1.0
end
