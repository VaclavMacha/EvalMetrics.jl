function mergesorted(x::Vector{T}, y::T) where T
    z, indexes = mergesorted(x, [y])
    return z, indexes[1]
end

function mergesorted(x::Vector{T}, y::Vector{S}) where {T,S}
    nx = length(x)
    ny = length(y)
    nz = nx + ny
    z  = Array{promote_type(T,S)}(undef, nz)

    ind_x   = 1
    ind_y   = 1
    indexes = ones(Int64, ny)

    @inbounds for ind_z in 1:nz
        val_x = x[ind_x]
        val_y = y[ind_y]
        if val_x <= val_y 
            z[ind_z]  = val_x
            ind_x    += 1
            if ind_x > nx
                indexes[ind_y:end] .= (ind_z + 1):nz
                z[(ind_z + 1):end] .= y[ind_y:end]
                break
            end
        else 
            z[ind_z]        = val_y
            indexes[ind_y]  = ind_z
            ind_y          += 1
            if ind_y > ny
                z[(ind_z+1):end] .= x[ind_x:end]
                break
            end
        end
    end
    return z, indexes
end