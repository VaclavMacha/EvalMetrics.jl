"""
    curve(fx::Function, fy::Function, x::AbstractVector{<:Counts})
    curve(fx, fy, target::IntegerVector, scores::RealVector, thresh::RealVector)
    curve(fx, fy, target::IntegerVector, scores::RealVector[, n::Int = length(target) + 1])

Calculates curve using function `fx` for `x` coordinates and `fy` for `y` coordinates.
Note that `fx`, `fy` must be functions of `Counts`.
"""
curve(fx::Function, fy::Function, x::AbstractVector{<:Counts}) =
    (fx.(x), fy.(x))
curve(fx, fy, target::IntegerVector, scores::RealVector, thresh::RealVector) =
    curve(fx, fy, counts(target, scores, thresh))
curve(fx, fy, target::IntegerVector, scores::RealVector, n::Int = length(target) + 1; kwargs...) =
    curve(fx, fy, counts(target, scores, thresholds(scores, n; kwargs...)))


"""
    auc(x::RealVector, y::RealVector)

Computes the area under curve `(x,y)` using trapezoidal rule.
"""
function auc(x::RealVector, y::RealVector)
    n   = length(x)
    val = zero(eltype(x))
    n == length(y) || throw(DimensionMismatch("Inconsistent lengths of `x` and `y`."))

    ind = indexin(unique(x), x)
    @views xu = x[ind]
    @views yu = y[ind]

    prm = sortperm(xu)

    @inbounds for i in 2:length(ind)
        Δx   = xu[prm[i]] - xu[prm[i-1]]
        fy   = yu[prm[i]] + yu[prm[i-1]]
        val += fy*Δx/2
    end
    return val
end


"""
    roc_curve(args...; kwargs...)

Calculates roc curve. See `curve` function for more details.
"""
roc_curve(args...; kwargs...) =
    curve(false_positive_rate, true_positive_rate, args...; kwargs...)


"""
    pr_curve(args...; kwargs...)

Calculates precision-recall curve. See `curve` function for more details.
"""
pr_curve(args...; kwargs...) =
    curve(recall, precision, args...; kwargs...)


"""
    pquant_curve(args...; kwargs...)

Calculates precision-quantile curve. See `curve` function for more details.
"""
pquant_curve(args...; kwargs...) =
    curve(quan, precision, args...; kwargs...)