# s = 2b-1
# b = (s+1)/2
# s = -1 <=> b = 0
# s = 1 <=> b = 1
function HOBO(ising::Ising{T}) where T<:Real
    variables = vars(ising)
    hobo = HOBO{T}(variables)
    for (key, val) = values(ising)
        if length(key) == 0
            hobo += HOBO(val, variables)
            continue
        end
        hobo += val*prod(2*variables[HOBO{T},k...]-one(T) for k=key)
    end
    hobo
end

function Ising(hobo::HOBO{T}) where T<:Real
    variables = vars(hobo)
    ising = Ising{T}(variables)
    for (key, val) = values(hobo)
        if length(key) == 0
            ising += Ising(val, variables)
            continue
        end
        ising += val*prod((variables[Ising{T},k...]+T(1))/T(2) for k=key)
    end
    ising
end
