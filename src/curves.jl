"""
    auc(x::RealVector, y::RealVector)

Computes the area under curve `(x,y)` using trapezoidal rule.
"""
function auc_trapezoidal(x::RealVector, y::RealVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("Inconsistent lengths of `x` and `y`."))

    if issorted(x)
        prm = 1:n 
    elseif issorted(x, rev = true)
        prm = n:-1:1
    else
        throw(ArgumentError("x must be sorted."))
    end

    val = zero(promote_type(eltype(x), eltype(y)))
    for i in 2:n
        Δx   = x[prm[i]]  - x[prm[i-1]]
        Δy   = (y[prm[i]] + y[prm[i-1]])/2
        if !(isnan(Δx) || isnan(Δy) || Δx == 0) 
            val += Δx*Δy
        end
    end
    return val
end

# TODO enable scattering points across pr and roc curves based on thresholds, tpr, fpr, or precision
# TODO implement curve persistance
# TODO implement multi-curve plotting
# TODO update readme with aucs and curve plotting

abstract type AbstractCurve end
struct PRCurve <: AbstractCurve end
struct ROCCurve <: AbstractCurve end

apply(::Type{PRCurve}, counts::CountVector) = (recall.(counts), precision.(counts))
apply(::Type{ROCCurve}, counts::CountVector) = (false_positive_rate.(counts), true_positive_rate.(counts))


function check_target(::Type{PRCurve}, enc::TwoClassEncoding, target::AbstractVector)
    if 0 == sum(ispositive.(enc, target))
        throw(ArgumentError("No positive samples present in `target`."))
    end
end


function check_target(::Type{ROCCurve}, enc::TwoClassEncoding, target::AbstractVector)
    if !(0 < sum(ispositive.(enc, target)) < length(target))
        throw(ArgumentError("Only one class present in `target`."))
    end
end


function curve_points(::Type{T}, enc::TwoClassEncoding, target::AbstractVector, scores::RealVector) where {T<:AbstractCurve}
    if length(scores) != length(target)
        throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    end
    check_target(T, enc, target)

    return ConfusionMatrix(enc, target, scores, thresholds(scores))
end


auroc(args...; kwargs...) = auc(ROCCurve, args...; kwargs...)
auprc(args...; kwargs...) = auc(PRCurve, args...; kwargs...)


auc(::Type{T}, args...) where {T<:AbstractCurve} = auc(T, current_encoding(), args...)

auc(::Type{T}, counts::CountVector) where {T<:AbstractCurve} = auc_trapezoidal(apply(T, counts)...)

auc(::Type{T}, enc::TwoClassEncoding, target::AbstractVector, scores::RealVector) where {T<:AbstractCurve} =
    auc(T, curve_points(T, enc, target, scores))


function auc(::Type{T}, enc::TwoClassEncoding, target::AbstractVector,
             scores::RealVector, thres::RealVector) where {T<:AbstractCurve}
    if length(scores) != length(target)
        throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    end
    check_target(T, enc, target)
    
    return auc(T, ConfusionMatrix(enc, target, scores, thres))
end


function auc_label(x, y, inpercent::Bool=false)
    val = auc_trapezoidal(x,y)
    if inpercent
        string("auc: ", round(100*val, digits = 2), "%")
    else
        string("auc: ", round(val, digits = 2))
    end
end

# function isdefault(plotattributes, key::Symbol)
#     !haskey(plotattributes, key) || plotattributes[key] == default(key)
# end

@recipe function f(::Type{Val{:mlcurve}}, x, y, z; indexes   = Int[],
                                                   aucshow   = true,
                                                   inpercent = true,
                                                   diagonal  = false)

    # Set attributes
    grid  --> true
    lims  --> (0, 1.01)

    # Add auc to legend
    if aucshow
        # if isdefault(plotattributes, :label)
        label := auc_label(x, y, inpercent)
        # else
        #     label := string(plotattributes[:label], " (", auc_label(x, y, inpercent), ")")
        # end
    end

    # main curve
    @series begin
        seriestype := :path
        marker     := :none
        x          := x
        y          := y
        ()
    end

    # points on the main curve
    @series begin
        primary           := false
        seriestype        := :scatter
        markerstrokecolor := :auto
        label             := ""
        x                 := x[indexes]
        y                 := y[indexes]
        ()
    end 

    # diagonal
    if diagonal
        @series begin
            primary    := false
            seriestype := :path
            fill       := false
            line       := (:red, :dash, 0.5)
            marker     := :none
            label      := ""
            x          := [0, 1]
            y          := [0, 1]
            ()
        end 
    end
end

@shorthands mlcurve

@recipe f(::Type{<:AbstractCurve}, x::AbstractArray, y::AbstractArray) = (x, y)

@recipe f(::Type{T}, target::AbstractVector, scores::RealVector) where {T<:AbstractCurve} =
    apply(T, curve_points(T, current_encoding(), target, scores))

@recipe f(::Type{T}, enc::TwoClassEncoding, target::AbstractVector, scores::RealVector) where {T<:AbstractCurve} =
    apply(T, curve_points(T, enc, target, scores))

@recipe f(::Type{T}, c::CountVector) where {T<:AbstractCurve} = apply(T, c)

@recipe f(::Type{T}, cs::AbstractArray{<:CountVector}) where {T<:AbstractCurve} = apply.(T, cs)


# ROC curve
@userplot ROCPlot

@recipe function f(h::ROCPlot)
    seriestype := :mlcurve
    diagonal   --> true
    legend     := :bottomright
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "ROC curve"
    xlabel     --> "false positive rate"
    ylabel     --> "true positive rate"

    (ROCCurve, h.args...)
end

# Plots.@deps ROCCurve


# Precision-Recall curve
@userplot PRPlot

@recipe function f(h::PRPlot)
    seriestype := :mlcurve
    legend     := :bottomleft
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "Precision-Recall curve"
    xlabel     --> "recall"
    ylabel     --> "precision"

    (PRCurve, h.args...)
end

# Plots.@deps PRCurve
