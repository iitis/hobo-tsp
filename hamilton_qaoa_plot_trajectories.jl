using PyPlot
using JLD2, FileIO
using LinearAlgebra
using SparseArrays
using Optim
using Statistics
using NPZ
using LineSearches
using Optim: minimizer
include("sparse_generator_loader.jl")
include("qaoa_utils.jl")

rc("text", usetex=true)
rc("font", family="serif", size=8)


cities_no = 3:5
k_max_list = [15, 15, 15]#fill(15, length(cities_no))
k_min_list = [1, 1, 1]#fill(15, length(cities_no))

repeating = 100

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

##

if "generate" ∈ ARGS
    for (city_no, k_min, k_max) = zip(cities_no, k_min_list, k_max_list)
        k_size = k_max-k_min+1
        println("######## city $city_no ##########")
        for mode = ["qubo", "hobo"]
            hamiltonian = npzread("hamilton/$(mode)_hamilton_$city_no.npz")
            ham_energy = reverse_hamiltonian_thr(hamiltonian, 0.01)
            n = length(hamiltonian)
            qubits_no = Int(log2(n))
            tmp_data = Dict("state" => zeros(ComplexF64, n),
                            "mul_vec" => zeros(ComplexF64, n),
                            "v" => zeros(ComplexF64, 2*n),
                            "tmp_vec" => zeros(ComplexF64, n),
                            "tmp_vec2" => zeros(ComplexF64, n),
                            "d" => load_sparsers(qubits_no))
            fg! = (F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data)
            results = zeros((k_size, 0))

            for m=1:repeating
                data_y = Float64[]
                filename = "hamilton/hamilton_results_$city_no-$m-$mode.jld2"
                if isfile(filename)
                    data = load(filename)
                    println("$city_no $m)")
                    probs = Float64[]
                    for el = vcat(data["small_results"], data["large_results"])
                        p = _energy_diffham!(ham_energy, hamiltonian, minimizer(el), tmp_data)
                        push!(probs, p)
                    end
                    results = hcat(results, probs)
                end
            end
            npzwrite("hamilton/results_$city_no-$mode.npz", results)
        end
    end
end


if "plot" ∈ ARGS
    #figure(figsize=[2.2,1.4])
    fig, axs = subplots(figsize=[6,1.4], nrows=1, ncols=3, sharex=true, sharey=true)
    k_min = 1
    k_max = 15
    for (city_ind, (ax, city_no)) = enumerate(zip(axs, cities_no))
        #figure(city_ind)
        println("######## city $city_no ##########")
        for (style,mode,label) = zip(["k.", "rx"], ["qubo", "hobo"], ["QUBO", "HOBO"])
            results = npzread("hamilton/results_$city_no-$mode.npz")[1:(k_max-k_min+1),:]
            data_x = (k_min - 1) .+ (1:size(results, 1))
            for m=1:size(results, 2)
                #plot(data_x, results[:,m], style,alpha=.5)
            end

            data_max = Float64[]
            for i=1:size(results,1)
                push!(data_max, maximum(results[i,:]))
            end
            ax.plot(data_x,  data_max, "-$(style)", label=label)
        end
        ##
        setp(axs[1,:], ylabel="probability of measuring\n in feasible space")
        setp(axs[:,1], xlabel="number of levels")
        yticks(0.:.5:1.)
        xticks([1,5,10,15])
        #setp(axs[:,1], xlabel="objective Ham. applied")

        xlim(k_min - .3, k_max + .3)
        if city_ind == 3
            ax.legend(loc=1, bbox_to_anchor = [1.62, 1.03])
        end

        ax.vlines(5, 0., 1., linestyle="--", linewidth=.5)
        ax.text(13.5, .5, "\\bf $('a'+city_ind-1))", fontweight="bold", fontsize=10)#, weight="bold")
    end
    savefig("plots/feasible_prob_hamilton_trajectories_multi.pdf", bbox_inches="tight")
end
