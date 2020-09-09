using LightGraphs

export
    max_cut,
    tsp_qubo,
    hamiltonian_cycle_qubo

#max cut
function max_cut(g::SimpleGraph, vars::Variables)
    @assert vars.indexcapacities == [nv(g), nv(g)]
    hobo = HOBO(vars)
    for edge = edges(g)
        v = edge.src
        w = edge.dst
        hobo -= (vars[v]-vars[w])^2
    end
    hobo
end

function max_cut(g::SimpleGraph; name::String="Max-Cut")
        max_cut(g, Variables(name, nv(g), "vertices"))
end

# hamiltonian cycle
function hamiltonian_cycle_qubo(n::Int, vars::Variables)
    @assert vars.indexcapacities == [n, n]
    hobo = HOBO(vars)

    for t=1:n
        hobo_tmp = HOBO(vars)
        for j=1:n
            hobo_tmp += vars[t, j]
        end
        hobo += (1. - hobo_tmp)^2
    end

    for j=1:n
        hobo_tmp = HOBO(vars)
        for t=1:n
            hobo_tmp += vars[t, j]
        end
        hobo += (1. - hobo_tmp)^2
    end

    #remove staying
    #for v=1:n, t=1:n
    #    hobo += vars[v, t]*vars[v, (t % n) + 1]
    #end
    hobo
end

function hamiltonian_cycle_qubo(n::Int, name::String="Hamiltonian Cycle Comlete Graph")
    hamiltonian_cycle_qubo(n, Variables(name, [n, n], ["timestep", "vertex"]))
end


function hamiltonian_cycle_qubo(g::SimpleGraph, vars::Variables)
    n = nv(g)
    hobo = hamiltonian_cycle_qubo(n, vars)

    #remove according to graph
    for edge = edges(LightGraphs.complement(g))
        u = edge.src
        v = edge.dst
        hobo += vars[t, u]*vars[t % n + 1, v]
        hobo += vars[t, v]*vars[t % n + 1, u]
    end

    hobo
end

function hamiltonian_cycle_qubo(g::SimpleGraph; name::String="Hamiltonian Cycle")
        hamiltonian_cycle_qubo(g, Variables(name, [nv(g),nv(g)], ["timestep", "vertex"]))
end

# tsp
function tsp_qubo(w::Matrix{T}, vars::Variables; B::Real=1., A::Real=2*B*maximum(w)) where T<:Real
    n = size(w, 1)
    @assert size(w, 1) == size(w, 2)
    @assert vars.indexcapacities == [n, n]
    @assert A >= 0 && B >= 0

    hobo_hamiltonian = hamiltonian_cycle_qubo(n, vars)
    hobo_tsp = HOBO(vars)
    #remove according to graph
    for u = 1:n, v = 1:n, t = 1:n
        if u != v
            hobo_tsp += w[u, v] * vars[t, u] * vars[t % n + 1, v]
        end
    end
    A*hobo_hamiltonian + B*hobo_tsp
end


function tsp_qubo(w::Matrix; name::String="TSP", B::Real=1., A::Real=2*B*maximum(w))
    tsp_qubo(w, Variables(name, collect(size(w)), ["timestep", "city"]), B=B, A=A)
end
