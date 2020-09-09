using BinaryOptimization
using Test
using LightGraphs
using Random
using LinearAlgebra


function simplified_tsp_qubo(w::Matrix{Float64})
    qubo = tsp_qubo(w)
    println(vars)
    n, _ = indexcapacities(vars(qubo))
    println(n)
    println(length(used_variables(qubo)))
    for i=1:(n-1)
        set_zero!(qubo, [1,i])
    end
    set_one!(qubo, [1,n])
    println(length(used_variables(qubo)))
    qubo
end

function simplified_tsp_hobo(w::Matrix{Float64})
    hobo = tsp_hobo(w)
    println(vars)
    n, logn = indexcapacities(vars(hobo))
    println(n)
    println(length(used_variables(hobo)))
    for i=1:logn
        set_zero!(hobo, [1,i])
    end
    println(length(used_variables(hobo)))
    hobo
end

n = 5
w = rand(n, n)
w += transpose(w) - 2*diagm(diag(w))
qubo = simplified_tsp_qubo(w)
hobo = simplified_tsp_hobo(w)

hamiltonian_qubo = bom_to_hamiltonian(qubo)
hamiltonian_hobo = bom_to_hamiltonian(hobo)
#pritnln(TravelingSalesmanExact.get_optimal_tour(w))
println(minimum(hamiltonian_qubo))
println(minimum(hamiltonian_hobo))
##
n = 5
variables = Variables("test", n)
hobo = variables[1]*variables[3] + 2*variables[1] -3 *variables[2] + 5
println(hobo)
println(used_variables(hobo))

n = 6
w = rand(n, n)
w += transpose(w) - 2*diagm(diag(w))

hobo_tsp = tsp_hobo(w)
println(used_variables(hobo_tsp))


##

n = 6
w = rand(n, n)
w += transpose(w) - 2*diagm(diag(w))



function perm_to_qubovar(v::Vector{Int})
    n = length(v)
    @assert length(unique(v)) == length(v) > 0
    @assert all(1 .<= v .<= n)
    #@assert dim(var) == 2
    #@assert indexcapacities(var)[1] == indexcapacities(var)[2] == n

    result = Set{Vector{Int}}()
    for i=1:n
        push!(result, [i,v[i]])
    end
    result
end

function perm_to_hobovar(v::Vector{Int})
    n = length(v)
    logn = ceil(Int, log(2, n))
    @assert length(unique(v)) == length(v) > 0
    @assert all(1 .<= v .<= n)

    result = Set{Vector{Int}}() # first to 0
    for t=1:n
        bits = reverse(digits(v[t]-1, base=2, pad=logn))
        for (ind, b) = filter(x->x[2]==1, collect(enumerate(bits)))
            push!(result, [t, ind])
        end
    end
    result
end

qubo = tsp_qubo(w, B=1., A=0.)
qubo_test = tsp_qubo(w, B=0., A=1.)
hobo = tsp_hobo(w, B=1., A=0.)
hobo_test = tsp_hobo(w, B=0., A=1.)

function right_value(w::Matrix{T}, v::Vector{Int}) where T<:Real
    n = length(v)
    @assert length(unique(v)) == length(v) > 0
    @assert all(1 .<= v .<= n)
    result = 0.
    for i=1:n
        result += w[v[i],v[i%n+1]]
    end
    result
end


#test qubo part
for _=1:100
    perm = Random.randperm(n)
    var_set_qubo = perm_to_qubovar(perm)
    var_set_hobo = perm_to_hobovar(perm)
    println("$perm $(right_value(w,perm))")
    println("$(qubo(var_set_qubo)) ($(qubo_test(var_set_qubo)))")
    println(var_set_hobo)
    println("$(hobo(var_set_hobo)) ($(hobo_test(var_set_hobo)))")
    @assert qubo(var_set_qubo) ≈ hobo(var_set_hobo)
end

println(hobo(Set{Vector{Int}}()))
