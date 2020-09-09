export
    AbstractVariable,
    Variables,
    name,
    indexcapacities,
    indexnames,
    dim,
    size,
    getindex


import Base: getindex, size
abstract type AbstractVariable end

rand(10_000,100_000)

struct Variables  <: AbstractVariable
    name::String
    indexcapacities::Vector{Int}
    indexnames::Vector{String}
    function Variables(name::String,
                       indexcapacities::Vector{Int},
                       indexnames::Vector{String}=fill("" for _ = indexcapacities))
        @assert length(name) > 0
        @assert length(indexnames) == length(indexcapacities)
        @assert all(i > 0 for i=indexcapacities)
        new(name, indexcapacities, indexnames)
    end
end

Variables(name::String, indcap::Int, indname=""::String) = Variables(name, [indcap], [indname])

name(vars::Variables) = vars.name
indexcapacities(vars::Variables) = vars.indexcapacities
indexnames(vars::Variables) = vars.indexnames

dim(vars::Variables) = length(indexnames(vars))
size(vars::Variables) = prod(vars.indexcapacities)

function getindex(var::Variables,
                  ::Type{T},
                  key::Vararg{Int}) where T<:BinaryOptimizationModel{S} where S<:Real
    if T<:Ising
        T(var, Dict{Vector{Vector{Int}},S}([[key...]]=>one(S)))
    elseif T<:HOBO
        T(var, Dict{Vector{Vector{Int}},S}([[key...]]=>one(S)))
    end
end

function getindex(var::Variables,
                  ::Type{T},
                  key::Vararg{Int}) where T<:BinaryOptimizationModel
    getindex(var, T{Float64}, key...)
end

getindex(var::Variables, key::Int...) = getindex(var, HOBO{Float64}, key...)


#function getindex(var::Variables, ind::Int)
    #@assert dim(var) == 1
    #HOBO(var, Dict([[ind]]=>1.))
#end
