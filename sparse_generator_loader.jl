using SparseArrays
using LinearAlgebra
using NPZ

function _1qubit_gen(qubit::Int, qubits_no::Int)
    n_left = 2^(qubit-1)
    n_right = 2^(qubits_no-qubit)
    eye_left = SparseMatrixCSC{ComplexF64,Int32}(I, n_left, n_left)
    eye_right = SparseMatrixCSC{ComplexF64,Int32}(I, n_right, n_right)
    x_rotation = SparseMatrixCSC{ComplexF64,Int32}(round.(exp(-1im*[0 1; 1 0])))

    if n_left < n_right
        kron(kron(eye_left, x_rotation), eye_right)
    else
        kron(eye_left, kron(x_rotation, eye_right))
    end
end

function generator(n::Int)
    @assert 32 >= n >= 1
    for k=1:n
        m = _1qubit_gen(k, n)
        if k == 1
            npzwrite("sparse_1qubitgate_data/colptr_$n.npz", m.colptr)
        end
        npzwrite("sparse_1qubitgate_data/rowval_$n-$k.npz", m.rowval)
        npzwrite("sparse_1qubitgate_data/topright_$n-$k.npz", Vector{Int32}(findall(x->x==-1im, m.nzval)))
    end
    nothing
end

function load_sparsers!(d::Dict, n::Int)
    @assert n >= 1
    @assert n ∉ keys(d)

    d["colptr"] = npzread("sparse_1qubitgate_data/colptr_$n.npz")
    for k=1:n
        d[k] = Dict{String,Vector}()
        d[k]["rowval"] = npzread("sparse_1qubitgate_data/rowval_$n-$k.npz")
        d[k]["topright"] = npzread("sparse_1qubitgate_data/topright_$n-$k.npz")
    end
    d
end

load_sparsers(n) = load_sparsers!(Dict(), n)

function sparse_up_1qubit!(v::Vector{ComplexF64},
                          n::Int,
                          k::Int,
                          angle::Float64,
                          d::Dict)
    fill!(v, ComplexF64(cos(angle)))
    @inbounds v[d[k]["topright"]] .= ComplexF64(-1im * sin(angle))
    SparseMatrixCSC{ComplexF64,Int32}(2^n, 2^n, d["colptr"], d[k]["rowval"], v)
end

function sparse_up_1qubit_dag!(v::Vector{ComplexF64},
                          n::Int,
                          k::Int,
                          angle::Float64,
                          d::Dict)
    fill!(v, ComplexF64(cos(angle)))
    @inbounds v[d[k]["topright"]] .= ComplexF64(1im * sin(angle))
    SparseMatrixCSC{ComplexF64,Int32}(2^n, 2^n, d["colptr"], d[k]["rowval"], v)
end

function sparse_up_1qubit_der!(v::Vector{ComplexF64},
                          n::Int,
                          k::Int,
                          angle::Float64,
                          d::Dict)
    fill!(v, ComplexF64(-sin(angle)))
    @inbounds v[d[k]["topright"]] .= -1im*ComplexF64(cos(angle))
    SparseMatrixCSC{ComplexF64,Int32}(2^n, 2^n, d["colptr"], d[k]["rowval"], v)
end

function sparse_up_1qubit_special!(v::Vector{ComplexF64},
                          n::Int,
                          k::Int,
                          angle::Float64,
                          d::Dict)
    fill!(v, zero(ComplexF64))
    @inbounds v[d[k]["topright"]] .= ComplexF64(-1im)
    SparseMatrixCSC{ComplexF64,Int32}(2^n, 2^n, d["colptr"], d[k]["rowval"], v)
end

function sparse_up_1qubit_der_dag!(v::Vector{ComplexF64},
                          n::Int,
                          k::Int,
                          angle::Float64,
                          d::Dict)
    fill!(v, ComplexF64(-sin(angle)))
    @inbounds v[d[k]["topright"]] .= 1im*ComplexF64(cos(angle))
    SparseMatrixCSC{ComplexF64,Int32}(2^n, 2^n, d["colptr"], d[k]["rowval"], v)
end
