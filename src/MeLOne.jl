module MeLOne

using LinearAlgebra

# Auxiliary
include("api.jl")
include("metrics.jl")

# Regression
include("linear-regression.jl")

# Classification
include("decision-tree.jl")
include("knn.jl")
include("logistic-regression.jl")
include("random-forest.jl")

end
