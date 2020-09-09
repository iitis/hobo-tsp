using Distributed

# directory

if length(ARGS) == 0 || ARGS[1] ∈ ["help", "-h", "--help"]
    println("\n\tjulia tsp_qaoa_experiment.jl dir_name no_experiments max_k repeating procs_no\n")
    exit()
end

dir_name = ARGS[1]
@assert isdir(dir_name)
# how many hobos/qubos
no_experiments = parse(Int, ARGS[2])
# maximum k
max_k = parse(Int, ARGS[3])
# how much each hamiltonian
repeating = parse(Int, ARGS[4])
# how many procs
addprocs(parse(Int, ARGS[5]))
# which_bunch
which_bunch = parse(Int, ARGS[6])

##
@everywhere using JLD2
@everywhere using FileIO
@everywhere using NPZ
@everywhere include("qaoa_optimizers.jl")

##

for _ in [1]
    function generate_experiment(mode::String, i::Int)
        println("######### experiment $i ($mode, $max_k) #########")
        kbreak = 5
        hamiltonian = npzread("$dir_name/$(mode)_$i.npz")
        upper = 10*pi
        results = Dict("experiment" => i,
                    "matrix_cost" => npzread("$dir_name/matrix_cost_$i.npz"),
                    "k" => max_k,
                    "mode" => mode,
                    "upper" => upper)
        d = load_sparsers(Int(log2(length(hamiltonian))))
        for m = 1:repeating
            filename = "$dir_name/$mode-exp$i-m$m-result.jld2"
            if isfile(filename)
                println("file exists")
                continue
            end
            results["small_results"] = qaoa(hamiltonian, kbreak-1, d=d, upper=upper)
            results["large_results"] = qaoa_trajectories_periodic(hamiltonian, max_k, kmin=kbreak, d=d, upper=upper)
            save(filename, results)
        end
        true
    end

    exp_tests = Iterators.product(["hobo", "qubo"], which_bunch .+ (0:4:96))

    pmap(i -> generate_experiment(i...), exp_tests)
end

rmprocs()
