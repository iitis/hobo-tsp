export
    set_zero!,
    set_one!,
    used_variables,
    size

function set_zero!(hobo::HOBO, v::Vector{Int})
    for k = keys(values(hobo))
        if v ∈ k
            delete!(values(hobo), k)
        end
    end
    hobo
end

function set_one!(hobo::HOBO, v::Vector{Int})
    for k = keys(values(hobo))
        if v ∈ k
            val = hobo[k...]
            new_k = setdiff(Set{Vector{Int}}(k), Set([v]))
            hobo[new_k...] += val
            delete!(values(hobo), k)
        end
    end
    hobo
end

function used_variables(hobo::BinaryOptimizationModel)
    sort(unique(vcat(keys(values(hobo))...)))
end
