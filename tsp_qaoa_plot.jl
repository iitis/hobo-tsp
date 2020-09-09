using Distributed
try
    addprocs(parse(Int, ARGS[1]))
catch
    nothing
end
##
using PyPlot
@everywhere using JLD2, FileIO
@everywhere using LinearAlgebra
@everywhere using SparseArrays
@everywhere using Optim
@everywhere using Statistics
@everywhere using NPZ
@everywhere include("sparse_generator_loader.jl")
@everywhere include("qaoa_optimizers.jl")


rc("text", usetex=true)
rc("font", family="serif", size=8)

cities_no = 3:5
k_max_list = [15, 15, 10]
k_min_list = [1, 1, 1]
experiments = 100
repeatings = [40, 40, 20]
## lowe eigenstate probability

function reverse_hamiltonian_thr(v::Vector{Float64}, thr::Float64)
    map(x -> x == 0 ? 1. : 0., v)
end

function up_low_std(data::Vector{Float64})
    the_mean = mean(data)
    up_std = filter(x-> x > the_mean, data)
    up_std = length(up_std) <= 2 ? 0 : sqrt(sum((up_std .- the_mean).^2)/length(up_std))
    low_std = filter(x-> x < the_mean, data)
    low_std = length(low_std) <= 2 ? 0 : sqrt(sum((low_std .- the_mean).^2)/length(low_std))
    the_mean, up_std, low_std
end

# hobo_6_result_14.jld2
function get_best(filename::String)
    d = load(filename)["results"]
    output = d[findmin(minimum.(d))[2]]
end

##

if "generate" ∈ ARGS
    for (city_no, k_min, k_max, repeating) = zip(cities_no, k_min_list, k_max_list, repeatings)
        println("######## city $city_no ##########")
        for mode = ["qubo", "hobo"]
            hamiltonian_hamilton = npzread("hamilton/$(mode)_hamilton_$city_no.npz")
            ham_feasible = reverse_hamiltonian_thr(hamiltonian_hamilton, 0.01)
            n = length(hamiltonian_hamilton)
            qubits_no = Int(log2(n))

            function generate_data(exp::Int)
                println("$city_no $mode $exp")
                tmp_data = Dict("state" => zeros(ComplexF64, n),
                                "mul_vec" => zeros(ComplexF64, n),
                                "v" => zeros(ComplexF64, 2*n),
                                "tmp_vec" => zeros(ComplexF64, n),
                                "tmp_vec2" => zeros(ComplexF64, n),
                                "d" => load_sparsers(qubits_no))

                hamiltonian_tsp = npzread("tsp_$city_no/$(mode)_$exp.npz")
                hamiltonian_tsp_feasible = hamiltonian_tsp .* ham_feasible
                true_min = minimum(collect(filter(x->x>0, hamiltonian_tsp_feasible)))
                true_max = maximum(collect(filter(x->x>0, hamiltonian_tsp_feasible)))

                fg! = (F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data)

                data_prob = zeros(k_max-k_min+1, repeating)
                data_energy = zeros(k_max-k_min+1, repeating)
                for m = 1:repeating
                    sol = load("tsp_$city_no/$(mode)-exp$exp-m$m-result.jld2")
                    results = vcat(sol["small_results"], sol["large_results"])
                    @assert length(results) >= k_max-k_min+1 && all(Optim.converged.(results))
                    for (ind, el) = enumerate(results)
                        #println(Optim.minimizer(el))
                        data_prob[ind, m] = _energy_diffham!(ham_feasible, hamiltonian_tsp, Optim.minimizer(el), tmp_data)

                        state = _state!(hamiltonian_tsp, Optim.minimizer(el), tmp_data)
                        state .= state .* ham_feasible
                        state .=  state ./ norm(state)
                        energy_state = sum(abs2.(state) .* hamiltonian_tsp_feasible)
                        data_energy[ind, m] = (energy_state - true_min)/(true_max - true_min)

                    end
                end
                npzwrite("tsp_$city_no/$mode-city$city_no-exp$exp-plotdata-prob.npz", data_prob)
                npzwrite("tsp_$city_no/$mode-city$city_no-exp$exp-plotdata-energy.npz", data_energy)
                return true
            end
            pmap(generate_data, 1:experiments)
        end
    end
end

if "plot" ∈ ARGS
    # probs
    fig, axs = subplots(figsize=[6,1.4], nrows=1, ncols=3, sharex=true, sharey=true)
    for (ax, city_no, k_min, k_max) = zip(axs, cities_no, k_min_list, k_max_list)
        println("######## city $city_no ##########")
        data_x = k_min:k_max
        for (color, mode, label) = zip(["k.", "rx"], ["qubo", "hobo"], ["QUBO", "HOBO"])
            data_y = zeros(length(data_x), 0)
            for exp = 1:experiments
                data_prob = npzread("tsp_$city_no/$mode-city$city_no-exp$exp-plotdata-prob.npz")
                data_y = hcat(data_y, mapslices(maximum, data_prob, dims=[2]))
            end
            ax.plot(data_x, mapslices(mean, data_y, dims=[2]), "-$color", label=label)
            ax.fill_between(k_min:k_max, mapslices(maximum, data_y, dims=[2])[:], mapslices(minimum, data_y, dims=[2])[:], color=color[[1]], alpha = 0.2)

            if city_no == 5
                ax.legend(loc=1, bbox_to_anchor = [1.62, 1.03])
            end
            ax.vlines(5, 0., 1., linestyle="--", linewidth=.5)
            ax.text(13.5, .5, "\\bf $('d'+city_no-3))", fontweight="bold", fontsize=10)#, weight="bold")
        end
    end
    setp(axs[1,:], ylabel="probability of measuring\n in feasible space")
    yticks(0.:.5:1.)
    xticks([1,5,10,15])
    setp(axs[:,1], xlabel="number of levels")
    savefig("plots/feasible_prob_tsp_trajectories_multi.pdf", bbox_inches="tight")
end
