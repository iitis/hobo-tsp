using Distributed

addprocs(parse(Int, ARGS[1]))

##
@everywhere using JLD2
@everywhere using FileIO
@everywhere using LineSearches
@everywhere using Optim
using NPZ
@everywhere include("qaoa_optimizers.jl")

##

repeating = parse(Int, ARGS[2])
kmax = parse(Int, ARGS[3])



for city_no = 3:5
    println("######### cities_no: $city_no #########")
    hamiltonian = Dict("qubo" => npzread("hamilton/qubo_hamilton_$city_no.npz"),
                       "hobo" => npzread("hamilton/hobo_hamilton_$city_no.npz"))
    upper = Dict("qubo" => 1. * pi, "hobo" => 2. * pi)

    function experiment_sample(m::Int, mode::String)
        kbreak = 5
        println("###### $mode $city_no $m #######")
        filename = "hamilton/hamilton_results_$city_no-$m-$mode.jld2"
        if isfile(filename)
            println("file exists")
            return "exists"
        end

        d = load_sparsers(Int(log2(length(hamiltonian[mode]))))
        results = Dict("kbreak" => kbreak,
                       "kmax" => kmax,
                       "upper" => upper[mode],
                       "mode" => mode,
                       "m" => m)
        results["small_results"] = qaoa(hamiltonian[mode], kbreak-1, d=d, upper=upper[mode])
        results["large_results"] = qaoa_trajectories_periodic(hamiltonian[mode], kmax, kmin=kbreak, d=d, upper=upper[mode])

        save(filename, results)
        return "done"
    end

    pmap(x-> experiment_sample(x...), collect(Iterators.product(1:repeating, ["hobo", "qubo"])))
end

rmprocs()
