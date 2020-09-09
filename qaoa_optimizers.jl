using LinearAlgebra
using SparseArrays
using Optim
using Random
using NPZ
using LineSearches
include("sparse_generator_loader.jl")
include("qaoa_utils.jl")

import Optim: retract!, project_tangent!

struct Periodic <: Manifold
    periods::Vector{Float64}
    function Periodic(v::Vector{Float64})
        @assert all(v .> 0.)
        new(v)
    end
end

periods(p::Periodic) = p.periods

function retract!(p::Periodic, x)
    per = periods(p)
    neg_ind = findall(x -> x<zero(0.), x)
    correctors_int = ceil.(Int, abs.(x[neg_ind] ./ per[neg_ind]))
    x[neg_ind] .+= per[neg_ind] .* correctors_int
    x .%= per
    x
end

project_tangent!(p::Periodic, g, x) = g

##
function qaoa(hamiltonian::Vector{T},
              kmax::Int;
              upper::Float64=Float64(pi),
              d::Dict=load_sparsers(Int(log2(length(hamiltonian))))) where T<:Real
    @assert kmax >= 1
    @assert upper > 0.

    n = length(hamiltonian)
    tmp_data = Dict("state" => zeros(ComplexF64, n),
                    "mul_vec" => zeros(ComplexF64, n),
                    "v" => zeros(ComplexF64, 2*n),
                    "tmp_vec" => zeros(ComplexF64, n),
                    "tmp_vec2" => zeros(ComplexF64, n),
                    "d" => d)
    fg! = Optim.only_fg!((F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data))
    opt = Optim.Options(g_tol=1e-5, allow_f_increases=true, iterations=5_000)

    results = Any[]
    for k = 1:kmax
        push!(results, 0)
        periods = vcat(fill(upper, k), fill(pi, k))
        optimizer = LBFGS(manifold=Periodic(periods), alphaguess=InitialStatic(alpha=0.001))

        converged = false
        while !converged
            init_times = rand(2*k) .* periods
            results[end] = optimize(fg!, init_times, optimizer, opt)
            if Optim.g_converged(results[end])
                converged = true
            end
        end
    end
    results
end

function qaoa_trajectories_periodic(hamiltonian::Vector{T},
                                    kmax::Int;
                                    upper::Float64=2*pi,
                                    kmin::Int=5,
                                    d::Dict=load_sparsers(Int(log2(length(hamiltonian))))) where T<:Real

    @assert kmax >= kmin >= 1
    @assert upper > 0.

    n = length(hamiltonian)
    tmp_data = Dict("state" => zeros(ComplexF64, n),
                    "mul_vec" => zeros(ComplexF64, n),
                    "v" => zeros(ComplexF64, 2*n),
                    "tmp_vec" => zeros(ComplexF64, n),
                    "tmp_vec2" => zeros(ComplexF64, n),
                    "d" => d)

    fg! = Optim.only_fg!((F, G, x) -> _fg!(F, G, hamiltonian, x, tmp_data))
    opt = Optim.Options(g_tol=1e-5, allow_f_increases=true, iterations=5_000)

    converged = false
    results = Any[]
    while !converged
        results = Any[]
        converged = true
        init_times = rand(2*kmin) .* vcat(fill(upper, kmin), fill(pi, kmin))

        for k = kmin:kmax
            periods = vcat(fill(upper, k), fill(pi, k))
            optimizer = LBFGS(manifold=Periodic(periods), alphaguess=InitialStatic(alpha=0.001))

            push!(results, optimize(fg!, init_times, optimizer, opt))
            if !Optim.g_converged(results[end])
                converged = false
                break
            end
            best_t = Optim.minimizer(results[end])
            init_times = vcat(best_t[1:k], best_t[k], best_t[k+1:end], best_t[end])
        end

    end
    println("DONE")
    results
end
