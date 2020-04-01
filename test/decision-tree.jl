using LinearAlgebra, Test

@testset "DecisionTree" begin
  N2 = MeLOne.Node(MeLOne.Subset([1,2,3], [1.0, 0.0]),
                   MeLOne.Subset([7,8], [0.0, 1.0]),
                   2, 70.0)
  N1 = MeLOne.Node(N2,
                   MeLOne.Subset([4,5,6], [0.0; 1.0]),
                   1, 15.0)
  model = DecisionTree(N1, [0, 1])

  X = [5 50; 10 40; 13 50; 18 40; 18 90; 20 65; 13 75; 10 80]
  y = [0, 0, 0, 1, 1, 1, 1, 1]

  @test predict(model, X) == y

  model = DecisionTree()
  fit!(model, X, y)
  @test predict(model, X) == y

  n = 400
  X = rand(n, 2)
  y = rand(0:1, n)
  model = DecisionTree()
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
  model = DecisionTree()
  fit!(model, X, y)
  X_te = [0.5 0.5;
          2.5 2.5;
          3.5 3.5]
  @test predict(model, X_te) == [:blue, :red, :red]
end
