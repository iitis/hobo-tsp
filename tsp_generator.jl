
using Distributed

if ARGS[1] == "--help" || ARGS[1] == "-h" || ARGS[1] == "help"
    println("\n\tFormat: julia tsp_generator.jl dirname n number [threads]

n is the number of cities, number is number of instances. dirname must not
exists. n should be small. Note (n-1)^2 variables are present for QUBO.
threads is the number of threads used, defaults to 1

Generated QUBO and HOBO assume that first cite is visted at time 1.

Following modules are needed: BinaryOptimization, TravelingSalesmanExact,
GLPK, NPZ, LinearAlgebra, Distributed")
    exit()
end

dir_out = ARGS[1]


n = parse(Int, ARGS[2])
@assert n >= 1

repeating = parse(Int, ARGS[3])
@assert repeating >= 1

if isdir(dir_out) || isfile(dir_out)
    error("$dir_out exists, please change or remove it")
    exit(1)
else
    mkdir(dir_out)
end

threads_no = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 0
addprocs(threads_no)



@everywhere using TravelingSalesmanExact, GLPK
@everywhere using LinearAlgebra
@everywhere set_default_optimizer!(GLPK.Optimizer)
@everywhere using BinaryOptimization
@everywhere using NPZ

@everywhere function solve_exact(w::Matrix{Float64})
    TravelingSalesmanExact.get_optimal_tour(w)
end

@everywhere function simplified_tsp_qubo(w::Matrix{Float64})
    qubo = tsp_qubo(w, A=2*maximum(w))
    n, _ = indexcapacities(vars(qubo))
    set_one!(qubo, [1,1])
    for i=2:n
        set_zero!(qubo, [1,i])
        set_zero!(qubo, [i,1])
    end
    qubo
end

@everywhere function simplified_tsp_hobo(w::Matrix{Float64})
    hobo = tsp_hobo(w, A=4*maximum(w))
    n, logn = indexcapacities(vars(hobo))
    for i=1:logn
        set_zero!(hobo, [1,i])
    end
    hobo
end

println("arguments are fine. Generating data started ($threads_no threads, $repeating samples)")
@sync @distributed for m = 1:repeating
    println("> $m ($repeating)")
    w = rand(n, n)
    w += transpose(w) - 2*diagm(diag(w))

    _, opt_cost = solve_exact(w)
    qubo = bom_to_hamiltonian(simplified_tsp_qubo(w))
    hobo = bom_to_hamiltonian(simplified_tsp_hobo(w))
    npzwrite("$dir_out/matrix_cost_$m.npz", w)
    npzwrite("$dir_out/qubo_$m.npz", qubo)
    npzwrite("$dir_out/hobo_$m.npz", hobo)

    if !(opt_cost ≈ minimum(qubo) ≈ minimum(hobo))
        println("It seems A and B are badly chosen ($opt_cost, $(minimum(qubo)) and $(minimum(hobo)) should be similar). Please send $m-th bunch to Adam Glos (aglos@iits.pl) if possible")
    end
end

rmprocs()
