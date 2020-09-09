module BinaryOptimization

export
    BinaryOptimizationModel,
    HOBO,
    Ising




include("binary_optimization_model.jl")
include("variables.jl")
include("ising_model.jl")
include("hobo_model.jl")
include("conversions.jl")
include("special_qubo.jl")
include("special_hobo.jl")
include("simplifiers.jl")
include("hamiltonians.jl")



#n = 10
#x = Variables("TSP", [10,10], ["time", "city"])
#getindex(x, 1, 2)

end # module
