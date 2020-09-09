export
    bom_to_hamiltonian

I(::Type{T}) where T<:Real = ones(T, 2)
Z(::Type{T}) where T<:Real = [one(T), -one(T)]

function _string_to_vec(T, s::String)
    result = [one(T)]
    for b = s
        if b == 'I'
            result = kron(result, I(T))
        elseif b == 'Z'
            result = kron(result, Z(T))
        else
            error("Bad string: chars have to be \'I\' or \'Z\'")
        end
    end
    result
end

function bom_to_hamiltonian(ising::Ising{T}) where T<:Real
    variables = used_variables(ising)
    degoffreedom = length(variables)

    result = zeros(T, 2^degoffreedom)
    for (key, val) = values(ising)
        pauli = ones(T, 1)
        for i=1:degoffreedom
            pauli = kron(pauli, variables[i] ∈ key ? Z(T) : I(T))
        end
        result += val * pauli
    end
    result
end

bom_to_hamiltonian(hobo::HOBO) = bom_to_hamiltonian(Ising(hobo))
