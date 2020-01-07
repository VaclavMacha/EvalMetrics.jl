# macro tools
splitwhere(ex::Expr)   = splitwhere(Val{ex.head}, ex)
splitwhere(ex::Symbol) = (ex, :Any)
splitwhere(::Type{Val{:(<:)}}, ex::Expr) = (ex.args[1], ex.args[2])


function true_type(arg::Expr, whereparams)
    T = splitarg(arg)[2]
    for wh in whereparams
        name, type = splitwhere(wh)
        if name == T
            return type
        end
    end
    return T
end


function pass_args(args)
    map(args) do arg
        name, type, is_splat, value = splitarg(arg)
        return is_splat ? Expr(:(...), name) : name 
    end
end


function pass_kwargs(kwargs)
    map(kwargs) do kw
        name, type, is_splat, value = splitarg(kw)
        return is_splat ? Expr(:(...), esc(name)) : Expr(:(=), esc(name), esc(name))
    end
end


# usermetric macro
macro usermetric(funcexpr::Expr)
    # split expression
    old        = splitdef(funcexpr) 
    old[:name] = esc(old[:name])

    # check type of the first argument
    T = true_type(old[:args][1], old[:whereparams])
    if !(eval(T) <: Counts)
        error("The first argument must be of type <:Counts got $(eval(T)).")
    end

    # extract name, args[2:end] and kwargs
    fname, args, kwargs = old[:name], old[:args][2:end], old[:kwargs]

    # function (target, predict, ...)
    new1        = copy(old)
    new1[:args] = vcat(:(target::IntegerVector),
                       :(predict::IntegerVector),
                       args[2:end])
    new1[:body] = quote 
        $(fname)(counts(target, predict), $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end

    # function (target, predict, ...)
    new2        = copy(old)
    new2[:args] = vcat(:(target::IntegerVector),
                       :(scores::RealVector),
                       :(threshold::Real),
                       args[2:end])
    new2[:body] = quote
        $(fname)(counts(target, scores, threshold), $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end

    # create final expresion
    quote
        @__doc__ $(combinedef(old))
        $(combinedef(new1))
        $(combinedef(new2))
    end
end


# merge sorted vectors
function mergesorted(x::Vector, y::Real)
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
                z[(ind_z + 1):end] .= ys[ind_y:end]
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


interpolate(y, x0, x1, y0, y1) = x0 + (y - y0)*(x1 - x0)/(y1 - y0)