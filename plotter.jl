using Pkg
Pkg.activate(".")

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

#figure(figsize=[2.2,1.4])
fig, axs = subplots(figsize=[6,3], nrows=2, ncols=3, sharex=true, sharey=true)
k_min = 1
k_max = 15
for (city_ind, city_no) = enumerate(cities_no)
    ax = axs[1,city_ind]
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
    
    setp(axs[2,:], xlabel="number of levels")
    yticks(0.:.5:1.)
    xticks([1,5,10,15])
    #setp(axs[:,1], xlabel="objective Ham. applied")

    xlim(k_min - .3, k_max + .3)
    if city_ind == 3
        ax.legend(loc=1, bbox_to_anchor = [1.62, 1.03])
    end

    ax.vlines(5, 0., 1., linestyle="--", linewidth=.5, color="k")
    ax.text(13.5, .5, "\\bf $('a'+city_ind-1))", fontweight="bold", fontsize=10)#, weight="bold")
end
setp(axs[1,1], ylabel=latexstring("prob. of measuring\n feasible solution \n\$W\\equiv 0\$"))
experiments = 100
k_max_list = [15, 15, 10]
k_min_list = [1, 1, 1]
repeatings = [40, 40, 20]

for (ind, (city_no, k_min, k_max)) = enumerate(zip(cities_no, k_min_list, k_max_list))
    ax=axs[2,ind]
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

        # if city_no == 5
        #     ax.legend(loc=1, bbox_to_anchor = [1.62, 1.03])
        # end
        ax.vlines(5, 0., 1., linestyle="--", linewidth=.5, color="k")
        ax.text(13.5, .5, "\\bf $('d'+city_no-3))", fontweight="bold", fontsize=10)#, weight="bold")
    end
end
setp(axs[2,1], ylabel=latexstring("prob. of measuring\n feasible solution \nrandom \$W\$"))
yticks(0.:.5:1.)
xticks([1,5,10,15])

savefig("plots/feasible_prob_trajectories_multi.pdf", bbox_inches="tight")