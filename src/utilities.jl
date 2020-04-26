get_ispos(classes::Tuple) = get_ispos(classes...)
get_ispos(neg::T, pos::T) where T <: LabelType = (x::T) -> x == pos
get_ispos(neg::Vector{T}, pos::T) where T <: LabelType = (x::T) -> x == pos
get_ispos(neg::T, pos::Vector{T}) where T <: LabelType = (x::T) -> x != neg

function get_ispos(neg::Vector{T}, pos::Vector{T}) where T <: LabelType
    if length(neg) <= length(pos) 
        return (x::T) -> !(x in neg)
    else
        return (x::T) -> x in pos
    end
end

get_classify(classes::Tuple) = get_classify(classes...)

function get_classify(neg::T, pos::T) where T <: LabelType 
    (s::Real, t::Real) -> s >= t ? pos : neg
end

function get_classify(neg::Vector{T}, pos::T) where T <: LabelType
    (s::Real, t::Real) -> s >= t ? pos : neg[1]
end

function get_classify(neg::T, pos::Vector{T}) where T <: LabelType
    (s::Real, t::Real) -> s >= t ? pos[1] : neg
end

function get_classify(neg::Vector{T}, pos::Vector{T}) where T <: LabelType
    (s::Real, t::Real) -> s >= t ? pos[1] : neg[1]
end


# -------------------------------------------------------------------------------
# Macro tools
# -------------------------------------------------------------------------------
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


function pass_args(arg::Symbol) 
    arg
end


function pass_args(args)
    map(args) do arg
        name, type, is_splat, value = splitarg(arg)
        return is_splat ? Expr(:(...), name) : name
    end
end


function pass_kwargs(kwarg::Symbol) 
    Expr(:(=), kwarg, kwarg)
end


function pass_kwargs(kwargs)
    map(kwargs) do kw
        name, type, is_splat, value = splitarg(kw)
        return is_splat ? Expr(:(...), name) : Expr(:(=), name, name)
    end
end


function make_kwarg(name, type, value)
    Expr(:kw, Expr(:(::), name, type), value)
end


# -------------------------------------------------------------------------------
# @usermetric macro
# -------------------------------------------------------------------------------
macro usermetric(funcexpr::Expr)
    # split expression
    old = splitdef(funcexpr) 

    # check type of the first argument
    T = true_type(old[:args][1], old[:whereparams])
    if !(eval(T) <: Counts)
        error("The first argument must be of type <:Counts got $(eval(T)).")
    end

    # create final expresion
    esc(quote
        Base.@__doc__ $(combinedef(old))
        $(create_type_1(old))
        $(create_type_2(old))
        $(create_type_3(old))
        $(create_type_4(old))
    end)
end

function create_type_1(old::Dict)
    fname, args, kwargs = old[:name], old[:args][2:end], old[:kwargs]

    new          = copy(old)
    new[:args]   = vcat(:(target::LabelVector), :(predict::LabelVector), args)
    new[:kwargs] = vcat(make_kwarg(:classes, :Tuple, :((0, 1))), kwargs)
    new[:body]   = quote
        x = counts(target, predict; $(pass_kwargs(:classes))) 
        $(fname)(x, $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end
    return combinedef(new)
end


function create_type_2(old::Dict)
    fname, args, kwargs = old[:name], old[:args][2:end], old[:kwargs]

    new          = copy(old)
    new[:args]   = vcat(:(target::LabelVector), :(scores::RealVector), :(thres::Real), args)
    new[:kwargs] = vcat(make_kwarg(:classes, :Tuple, :((0, 1))), kwargs)
    new[:body]   = quote
        x = counts(target, scores, thres; $(pass_kwargs(:classes)))
        $(fname)(x, $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end
    return combinedef(new)
end


function create_type_3(old::Dict)
    fname, args, kwargs = old[:name], old[:args][2:end], old[:kwargs]

    new        = copy(old)
    new[:args] = vcat(:(x::Vector{Counts}), args)
    new[:body] = quote
        $(fname).(x, $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end
    return combinedef(new)
end


function create_type_4(old::Dict)
    fname, args, kwargs = old[:name], old[:args][2:end], old[:kwargs]

    new          = copy(old)
    new[:args]   = vcat(:(target::LabelVector), :(scores::RealVector), :(thres::RealVector), args)
    new[:kwargs] = vcat(make_kwarg(:classes, :Tuple, :((0, 1))), kwargs)
    new[:body]   = quote
        x = counts(target, scores, thres; $(pass_kwargs(:classes)))
        $(fname).(x, $(pass_args(args)...); $(pass_kwargs(kwargs)...))
    end
    return combinedef(new)
end

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
