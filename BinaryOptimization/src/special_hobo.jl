
export
    tsp_hobo


# functions below assumes that hobos accepts ony one or zero

function are_same(hobos::Vector{HOBO{T}}, n::Int) where T<:Real
    bit_length = length(hobos)
    bits = reverse(digits(n, base=2, pad=bit_length))
    @assert length(bits) == bit_length
    reduce(and, [b == 0 ? one(T) - h : h  for (h,b)=zip(hobos, bits)])
end

function are_different(v1::Vector{HOBO{T}}, v2::Vector{HOBO{T}}) where T<:Real
    neg(reduce(or, xor.(v1, v2)))
end

# assumes numbering from 0 and that the vectors of HOBO
# are of the form var[...]
function smaller_equal_than(hobos::Vector{HOBO{T}}, n::Int) where T<:Real
    k = length(hobos)
    # no point for n = 2^k-1 or n = 0
    @assert n+1 == 2^k || 1 <= n <= 2^k-2
    bits = reverse(digits(n, base=2))
    zero_pos = findall(x->x==0, bits)
    @assert bits[1] == 1
    @assert length(bits) == k

    result = zero(T)*hobos[1]
    for ind=zero_pos
        result_tmp = prod(b==0 ? one(T) - h : h for (h,b)=zip(hobos, bits[1:ind-1]))
        result += result_tmp*hobos[ind]
    end
    result
end

function _hamiltonian_cycle_complete_hobo(vars::Variables)
    n, logn = indexcapacities(vars)
    hobo = HOBO(vars)

    # prohibit numbers
    for t = 1:n
        hobo += smaller_equal_than([vars[t,i] for i=1:logn], n-1)
    end

    # different for different time
    for t1 = 1:n, t2 = (t1+1):n
        hobo += are_different([vars[t1,i] for i=1:logn], [vars[t2,i] for i=1:logn])
    end
    hobo
end

function _hobo_tsp_part(w::Matrix{T}, vars::Variables) where T<:Real
    n = size(w, 1)
    logn = ceil(Int, log(2, n))

    @assert size(w, 2) == n
    hobo = HOBO(vars)

    # prohibit numbers
    for t=1:n, u=1:n, v=1:n
        if u != v
            hobo_tmp = are_same([vars[t,i] for i=1:logn], u-1)
            hobo_tmp *= are_same([vars[t%n+1,i] for i=1:logn], v-1)
            hobo += w[u,v]*hobo_tmp
        end
    end
    hobo
end

function tsp_hobo(w::Matrix{T}, vars::Variables; B::Real=1., A::Real=2*B*maximum(w)) where T<:Real
    n = size(w, 1)
    logn = ceil(Int, log(2, n))

    @assert size(w, 2) == n
    @assert indexcapacities(vars) == [n, logn]
    @assert A >= zero(T) && B >= zero(T)

    A*_hamiltonian_cycle_complete_hobo(vars) + B*_hobo_tsp_part(w, vars)
end

function tsp_hobo(w::Matrix; name::String="TSP", B::Real=1., A::Real=2*B*maximum(w))
    n = size(w, 1)
    tsp_hobo(w, Variables(name, [n, ceil(Int, log(2, n))], ["timestep", "city_binary"]), B=B, A=A)
end
