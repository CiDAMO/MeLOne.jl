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
end
