using LinearAlgebra
using SparseArrays
using Optim
using Random
using NPZ
include("sparse_generator_loader.jl")

##
function _state!(hamiltonian::Vector{Float64}, times::Vector{Float64}, tmp_data::Dict)
    k = div(length(times), 2)
    qubits_no = Int(log2(length(hamiltonian)))

    state = tmp_data["state"]
    mulvec = tmp_data["mul_vec"]


    state .= ComplexF64(1/sqrt(length(hamiltonian)))

    for (p, r) = zip(times[1:k], times[(k+1):end])
        @inbounds broadcast!((x,y) -> x*exp(-1im*p*y), state, state, hamiltonian)
        for q = 1:qubits_no
            @inbounds mul!(mulvec, sparse_up_1qubit!(tmp_data["v"], qubits_no, q, r, tmp_data["d"]), state)
            #@inbounds state .= sparse_up_1qubit!(tmp_data["v"], qubits_no, q, r, tmp_data["d"])*state
            tmp = mulvec
            mulvec = state
            state = tmp
        end
    end
    state
end

function _energy!(hamiltonian::Vector{Float64}, times::Vector{Float64}, tmp_data::Dict)
    state = _state!(hamiltonian, times, tmp_data)
    @inbounds broadcast!((x,y) -> abs2(x)*y, state, state, hamiltonian)
    abs(sum(state))
end

function _energy_diffham!(ham_energy::Vector{Float64},
                           hamiltonian::Vector{Float64},
                           times::Vector{Float64},
                           tmp_data::Dict)
    state = _state!(hamiltonian, times, tmp_data)
    @inbounds broadcast!((x,y) -> abs2(x)*y, state, state, ham_energy)
    abs(sum(state))
end

function _fg!(F, gradient, hamiltonian::Vector{Float64}, times, tmp_data::Dict, upper::Float64=1.)
    k = div(length(times), 2)
    qubits_no = Int(log2(length(hamiltonian)))
    tmp_vec = tmp_data["tmp_vec"]
    tmp_vec2 = tmp_data["tmp_vec2"]
    d = tmp_data["d"]
    v = tmp_data["v"]

    state = _state!(hamiltonian, times, tmp_data)
    mulvec = state === tmp_data["mul_vec"] ? tmp_data["state"] : tmp_data["mul_vec"]

    copy!(tmp_vec, state)
    @inbounds broadcast!(*, state, hamiltonian, state)
    if F != nothing
        @inbounds broadcast!((x,y) -> conj(x)*y, mulvec, state, tmp_vec)
        F = abs(sum(mulvec))
    end
    if gradient != nothing
        for (i, p, r) = zip(k:-1:1, times[k:-1:1], times[2*k:-1:(k+1)])
            tmp_vec2 .= 0.
            for q = 1:qubits_no
                #@inbounds tmp_vec .= sparse_up_1qubit_dag!(v, qubits_no, q, r, d)*tmp_vec
                #@inbounds tmp_vec .= sparse_up_1qubit_der!(v, qubits_no, q, r, d)*tmp_vec
                #@inbounds tmp_vec .= sparse_up_1qubit_special!(v, qubits_no, q, r, d)*mulvec
                @inbounds mul!(mulvec, sparse_up_1qubit_special!(v, qubits_no, q, r, d), tmp_vec)
                @inbounds broadcast!((x, y, z) -> x+conj(y)*z, tmp_vec2, tmp_vec2, mulvec, state)
                # one can use dot: the differnce is neglible
                #println(sum(tmp_vec2))
                #@inbounds tmp_vec .= sparse_up_1qubit_der_dag!(v, qubits_no, q, r, d)*tmp_vec
                @inbounds mul!(tmp_vec, sparse_up_1qubit_der_dag!(v, qubits_no, q, r, d), mulvec)

                #@inbounds state .= sparse_up_1qubit_dag!(v, qubits_no, q, r, d)*state
                @inbounds mul!(mulvec, sparse_up_1qubit_dag!(v, qubits_no, q, r, d), state)
                tmp = mulvec
                mulvec = state
                state = tmp
            end
            gradient[k+i] = 2*real(sum(tmp_vec2))

            tmp_vec2 .= 0.
            @inbounds broadcast!((x, y) -> exp(1im*p*y)*x, tmp_vec, tmp_vec, hamiltonian)
            @inbounds broadcast!((x, y) -> exp(1im*p*y)*x, state, state, hamiltonian)
            @inbounds broadcast!((x, y, z) -> 1im*conj(x)*y*z,
                                 tmp_vec2, tmp_vec, hamiltonian, state)
            gradient[i] = 2*real(sum(tmp_vec2))
        end
    end
    F
end
