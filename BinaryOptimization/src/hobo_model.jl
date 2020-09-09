export
    HOBO

import Base: *

"""
HOBO classes. Current HOBO object should not contain zero(T) values, which may result
in unnecessary memory blow.
"""
struct HOBO{T} <: BinaryOptimizationModel{T}
    vars::Variables
    values::Dict{Vector{Vector{Int}},T}
    function HOBO{T}(vars::Variables,
                     values::Dict{Vector{Vector{Int}},T}) where T<:Real
        final_values = Dict{Vector{Vector{Int}},T}()

        for (key, val) = values
            final_key = clean_key(HOBO, key)
            @assert all(length(i) == dim(vars) for i = key)
            @assert all(all(0 .< ind .<= indexcapacities(vars)) for ind=final_key)
            _dict_add!(final_values, final_key, val)
        end
        new{T}(vars, final_values)
    end
end

function HOBO(::Type{T}, vars::Variables) where T
    HOBO{T}(vars, Dict{Vector{Vector{Int}},T}())
end

HOBO(vars::Variables) = HOBO(Float64, vars::Variables)
HOBO{T}(vars::Variables) where T<:Real = HOBO(T, vars::Variables)

function HOBO(num::T, vars::Variables) where T<:Real
    HOBO{T}(vars, Dict(Vector{Int}[] => num))
end

clean_key(::Type{HOBO}, key::Vector{Vector{Int}}) = sort(unique(key))
clean_key(::Type{HOBO{T}}, key::Vector{Vector{Int}}) where T<:Real = clean_key(HOBO, key)

## may be improved - smart tree walker
function (hobo::HOBO{T})(keys_list::Set{Vector{Int}}) where T<:Real
    result = zero(T)
    for (key, val) = values(hobo)
        if all(k ∈ keys_list for k = key)
            result += val
        end
    end
    result
end

#(hobo::HOBO{<:Real})(keys_list) = (hobo::HOBO)(Set(keys_list...))
#(hobo::HOBO{<:Real})(keys_list::Vector{Vector{Int}}) = (hobo::HOBO)([keys_list])

## logical functions

or(hobo1::HOBO, hobo2::HOBO) = hobo1 + hobo2 - hobo1*hobo2
xor(hobo1::HOBO{T}, hobo2::HOBO{T}) where T<:Real = hobo1 + hobo2 - T(2)*hobo1*hobo2
and(hobo1::HOBO, hobo2::HOBO) = hobo1 * hobo2
neg(hobo::HOBO{T}) where T = one(T) - hobo
