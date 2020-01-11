# -------------------------------------------------------------------------------
# ispos function
# -------------------------------------------------------------------------------
const LabelType                 = Union{Bool, Real, String, Symbol}
const LabelVector{T<:LabelType} = AbstractArray{T,1}

get_ispos(classes::Tuple) = get_ispos(classes...)

function get_ispos(neg::LabelType, pos::LabelType)
    ispos(x::LabelType) = x == pos
end

function get_ispos(neg::LabelVector{T}, pos::S) where {T<:LabelType, S<:LabelType} 
    ispos(x::LabelType) = x == pos
end

function get_ispos(neg::T, pos::LabelVector{S}) where {T<:LabelType, S<:LabelType} 
    ispos(x::LabelType) = x != neg
end

function get_ispos(neg::LabelVector, pos::LabelVector)
    if length(neg) <= length(pos) 
        _get_ispos1(neg, pos)
    else
        _get_ispos1(neg, pos)
    end
end

function _get_ispos1(neg::LabelVector, pos::LabelVector)
    ispos(x::LabelType) = !(x in neg)
end

function _get_ispos2(neg::LabelVector, pos::LabelVector)
    ispos(x::LabelType) =  x in pos
end


# -------------------------------------------------------------------------------
# classify function
# -------------------------------------------------------------------------------
get_classify(classes::Tuple) =
    get_classify(classes...)

get_classify(neg::LabelType, pos::LabelType) = 
    (s::Real, t::Real) -> s >= t ? pos : neg

get_classify(neg::LabelVector, pos::LabelType) = 
    (s::Real, t::Real) -> s >= t ? pos : neg[1]

get_classify(neg::LabelType, pos::LabelVector) = 
    (s::Real, t::Real) -> s >= t ? pos[1] : neg

get_classify(neg::LabelVector, pos::LabelVector) = 
    (s::Real, t::Real) -> s >= t ? pos[1] : neg[1]

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
        @__doc__ $(combinedef(old))
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
    new[:args] = vcat(:(x::CountsArray), args)
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


# -------------------------------------------------------------------------------
# merge sorted vectors
# -------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------
# area under the curve
# -------------------------------------------------------------------------------
"""
    auc(x::RealVector, y::RealVector)

Computes the area under curve `(x,y)` using trapezoidal rule.
"""
function auc(x::RealVector, y::RealVector)
    n   = length(x)
    val = zero(eltype(x))
    n == length(y) || throw(DimensionMismatch("Inconsistent lengths of `x` and `y`."))

    if issorted(x)
        prm = 1:n
    else
        prm = sortperm(x)
    end

    @inbounds for i in 2:n
        Δx   = x[prm[i]]  - x[prm[i-1]]
        Δy   = (y[prm[i]] + y[prm[i-1]])/2
        
        if !(isnan(Δx) || isnan(Δy))
            val += Δx*Δy
        end
    end
    return val
end