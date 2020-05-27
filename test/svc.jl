using LinearAlgebra, Test

@testset "SVC" begin
  X = [0.0 0.0;
       0.5 0.0;
       0.0 0.5;
       2.0 2.0;
       3.0 3.0;
       4.0 5.0]
  y = [1, 1, 1, -1, -1, -1]
  model = SVC()
  fit!(model, X, y)
  @test predict(model, X) == y

  n = 50
  X = randn(n, 2)
  y = [X[i,1]^2 + X[i,2]^2 > 1.4 ? 1 : -1 for i = 1:n]
  model = SVC(kernel=:rbf, C=1e6, gamma=1.0, coef0=1.0, degree=4)
  fit!(model, X, y)
  y_pred = predict(model, X)
  @test accuracy_score(y, y_pred) == 1.0
end
