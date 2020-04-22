module MeLOne

using LinearAlgebra

# Auxiliary
include("api.jl")
include("metrics.jl")

# Regression
include("linear-regression.jl")
include("knn-regressor.jl")

# Classification
include("decision-tree.jl")
include("knn-classifier.jl")
include("logistic-regression.jl")
include("random-forest.jl")

end
