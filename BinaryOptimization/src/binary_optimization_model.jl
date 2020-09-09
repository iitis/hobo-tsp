export
    BinaryOptimizationModel,
    vars,
    values,
    order,
    isquadratic

import Base: *, /, +, -, ^, getindex, setindex!

abstract type BinaryOptimizationModel{T<:Real} end

vars(bqm::BinaryOptimizationModel) = bqm.vars
values(bqm::BinaryOptimizationModel) = bqm.values

function _dict_add!(dict::Dict{Vector{Vector{Int}},T},
                    index::Vector{Vector{Int}},
                    val::Real) where T<:Real
    if index ∈ keys(dict)
        dict[index] += val
    else
        dict[index] = val
    end
    if dict[index] == zero(T)
        pop!(dict, index)
    end
    dict
end

function _dict_add_no_check!(dict::Dict{Vector{Vector{Int}},T},
                             index::Vector{Vector{Int}},
                             val::Real) where T<:Real
    if index ∈ keys(dict)
        dict[index] += val
    else
        dict[index] = val
    end
    dict
end

function getindex(bqm::S, key...) where S<:BinaryOptimizationModel{T} where T<:Real
    if isempty(key)
        key = Vector{Int}[]
    end
    the_key = clean_key(S, collect(key))
    if the_key ∈ keys(values(bqm))
        values(bqm)[the_key]
    else
        zero(T)
    end
end

function setindex!(bqm::S,
                   val::Real,
                   key...) where S<:BinaryOptimizationModel{T} where T<:Real
    if isempty(key)
        key = Vector{Int}[]
    end
    the_key = clean_key(S, collect(key))
    if val != zero(T)
        values(bqm)[the_key] = T(val)
    elseif the_key ∈ keys(values(bqm))
        delete!(values(bqm), the_key)
    end
    bqm
end

## Multiplication by number
function *(bqm::S, var::T) where {S<:BinaryOptimizationModel{T} where T<:Real, T<:Real}
    bqm_copy = S(vars(bqm), copy(values(bqm)))
    if var == zero(T)
        empty!(values(bqm_copy))
        return bqm_copy
    elseif var == one(T)
        return bqm_copy
    else
        for k = keys(values(bqm_copy))
            bqm_copy[k...] *= var
        end
    end
    bqm_copy
end

*(var::Real, bqm::BinaryOptimizationModel) = *(bqm, var)
/(bqm::BinaryOptimizationModel, var::T) where T<:Real = *(bqm, one(T)/var)
-(bqm::BinaryOptimizationModel{T}) where T = *(bqm, -one(T))
^(bqm::BinaryOptimizationModel, n::Int) = n == 1 ? bqm : bqm*bqm^(n-1)

## sum with number
function +(bqm::S, var::T) where {S<:BinaryOptimizationModel{T} where T<:Real,T<:Real}
    bqm_copy = S(vars(bqm), copy(values(bqm)))
    empty_array = Vector{Int}[]
    _dict_add!(values(bqm_copy), Vector{Int}[], var)
    bqm_copy
end

+(var::Real, bqm::BinaryOptimizationModel) = +(bqm, var)
-(bqm::BinaryOptimizationModel, var::Real) = +(bqm, -var)

## sum of bqm
function _bqm_add(bqm1::S, bqm2::S) where S<:BinaryOptimizationModel{<:Real}
    bqm_result = S(vars(bqm1), copy(values(bqm1)))
    for (key, val) = values(bqm2)
        _dict_add!(values(bqm_result), key, val)
    end
    bqm_result
end

function +(bqm1::BinaryOptimizationModel, bqm2::BinaryOptimizationModel)
    if length(values(bqm1)) > length(values(bqm2))
        _bqm_add(bqm1, bqm2)
    else
        _bqm_add(bqm2, bqm1)
    end
end

-(bqm1::BinaryOptimizationModel, bqm2::BinaryOptimizationModel) = bqm1 + (-bqm2)
-(var::Float64, bqm::BinaryOptimizationModel) = var + (-bqm)

## Multiplication of hobos
function *(bqm1::S, bqm2::S) where S<:BinaryOptimizationModel{T} where T<:Real
    @assert vars(bqm1) == vars(bqm2)
    bqm_result = S(vars(bqm1))
    for (key1, val1) = values(bqm1), (key2, val2) = values(bqm2)
        _dict_add!(values(bqm_result), clean_key(S, vcat(key1, key2)), val1*val2)
    end
    bqm_result
end

##
function order(bqm::BinaryOptimizationModel)
    if !isempty(values(bqm))
        maximum(length.(keys(values(bqm))))
    else
        0
    end
end

isquadratic(bqm::BinaryOptimizationModel) = order(bqm) <= 2
