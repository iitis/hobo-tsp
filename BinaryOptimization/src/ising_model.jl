export
    Ising

import Base: *

"""
HOBO classes. Current HOBO object should not contain zero(T) values, which may result
in unnecessary memory blow.
"""
struct Ising{T} <: BinaryOptimizationModel{T}
    vars::Variables
    values::Dict{Vector{Vector{Int}},T}
    function Ising{T}(vars::Variables,
                      values::Dict{Vector{Vector{Int}},T}) where T<:Real
        final_values = Dict{Vector{Vector{Int}},T}()

        for (key, val) = values
            final_key = clean_key(Ising, key)
            @assert all(length(i) == dim(vars) for i = key)
            @assert all(all(0 .< ind .<= indexcapacities(vars)) for ind=final_key)
            _dict_add!(final_values, final_key, val)
        end
        new{T}(vars, final_values)
    end
end

function Ising(::Type{T}, vars::Variables) where T
    Ising{T}(vars, Dict{Vector{Vector{Int}},T}())
end

Ising(vars::Variables) = Ising(Float64, vars::Variables)
Ising{T}(vars::Variables) where T<:Real = Ising(T, vars::Variables)

function Ising(num::T, vars::Variables) where T<:Real
    Ising{T}(vars, Dict(Vector{Int}[] => num))
end

function clean_key(::Type{Ising}, key::Vector{Vector{Int}})
    unique_keys = unique(key)
    result = Vector{Int}[]
    for k in unique_keys
        if isodd(count(x -> x == k, key))
            push!(result, k)
        end
    end
    sort(result)
end

clean_key(::Type{Ising{T}}, key::Vector{Vector{Int}}) where T<:Real = clean_key(HOBO, key)

## may be improved - smart tree walker
function (ising::Ising{T})(keys_list::Set{Vector{Int}}) where T<:Real
    result = zero(T)
    for (key, val) = values(ising)
        if isodd(length(setdiff(Set(key), Set(keys_list))))
            result -= val
        else
            result += val
        end
    end
    result
end

#(hobo::HOBO{<:Real})(keys_list) = (hobo::HOBO)(Set(keys_list...))
#(hobo::HOBO{<:Real})(keys_list::Vector{Vector{Int}}) = (hobo::HOBO)([keys_list])
