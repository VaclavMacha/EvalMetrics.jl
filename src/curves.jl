# TODO enable scattering points across pr and roc curves based on thresholds, tpr, fpr, or precision

abstract type AbstractCurve end

# by default always compute auc and curve points from all possible points
function auc(::Type{C}, args...; kwargs...) where {C<:AbstractCurve}
    auc_trapezoidal(curve(C, args...; kwargs...))
end
function auc(::Type{C}, enc::TwoClassEncoding, targets::AbstractArray{<:AbstractVector},
               scores::AbstractArray{<:RealVector}, args...; kwargs...) where {C<:AbstractCurve}
    auc_trapezoidal.(curve(C, enc, targets, scores, args...; kwargs...))
end
function auc(::Type{C}, targets::AbstractArray{<:AbstractVector},
               scores::AbstractArray{<:RealVector}, args...; kwargs...) where {C<:AbstractCurve}
    auc_trapezoidal.(curve(C, targets, scores, args...; kwargs...))
end
function curve(::Type{C}, args...; npoints=Inf) where {C<:AbstractCurve}
    apply(C, args...; npoints=npoints)
end
function curve(::Type{C}, cs::CMVector, args...; kwargs...) where {C<:AbstractCurve}
    apply(C, cs, args...)
end

macro curve(name)
    name_lw = Symbol(lowercase(string(name)))
    name_auc = Symbol(lowercase(string("au_", name)))

    quote 
        abstract type $(esc(name)) <: AbstractCurve end

        Base.@__doc__  function $(esc(name_lw))(args...; kwargs...) 
            curve($(esc(name)), args...; kwargs...)
        end

        function $(esc(name_auc))(args...; kwargs...) 
            auc($(esc(name)), args...; kwargs...)
        end
    end
end


@recipe function f(::Type{Val{:mlcurve}}, x, y, z; indexes=Int[], diagonal=false)
    # main curve
    @series begin
        seriestype := :path
        marker     := :none
        x          := x
        y          := y
        ()
    end

    # points on the main curve
    if !isempty(indexes)
        @series begin
            primary           := false
            seriestype        := :scatter
            markerstrokecolor := :auto
            label             := ""
            x                 := x[indexes]
            y                 := y[indexes]
            ()
        end 
    end

    # diagonal
    if diagonal && get(plotattributes, :xscale, :identity) === :identity
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

apply(::Type{C}, targets, scores; kwargs...) where {C<:AbstractCurve} =
    apply(C, current_encoding(), targets, scores; kwargs...)
apply(::Type{C}, targets, scores, thres; kwargs...) where {C<:AbstractCurve} =
    apply(C, current_encoding(), targets, scores, thres; kwargs...)

function apply(::Type{C}, enc::TwoClassEncoding,
               targets::AbstractArray{<:AbstractVector},
               scores::AbstractArray{<:RealVector}, args...; kwargs...)  where {C<:AbstractCurve}
    return [apply(C, t, s, args...; kwargs...) for (t,s) in zip(targets, scores)]
end

function apply(::Type{C}, enc::TwoClassEncoding,
               targets::AbstractVector, scores::RealVector;
               npoints::Real=300, xscale::Symbol=:identity,
               xlims=(0, 1), kwargs...) where {C<:AbstractCurve}
    # maximal resolution
    if npoints >= length(targets) + 1
        thres = thresholds(scores)
    else
        if xscale === :identity
            quantils = range(0, stop=1.0, length=npoints)
        else
            # if user didn't provide custom xlims, compute them so that we sample efficiently around
            # the area with lowest nonzero FPR/TPR
            if xlims == (0, 1)
                xlims = (lowest_metric_value(C, enc, targets), 1)
            end
            quantils = exp10.(range(log10(xlims[1]), log10(min(1, xlims[2])), length=npoints))
        end
        thres = sampling_function(C)(enc, targets, scores, quantils)
    end
    return apply(C, enc, targets, scores, thres)
end


function apply(::Type{C},
               enc::TwoClassEncoding,
               targets::AbstractVector,
               scores::RealVector, thres::RealVector; kwargs...)  where {C<:AbstractCurve}
    if !(0 < sum(ispositive.(enc, targets)) < length(targets))
        throw(ArgumentError("Only one class present in `targets` with encoding $enc."))
    end
    return apply(C, ConfusionMatrix(enc, targets, scores, thres))
end

@recipe function f(::Type{C}, args...) where {C<:AbstractCurve}
    points = apply(C, args...; plotattributes...)
    xl, yl = _lims(points, plotattributes)
    xlims --> xl
    ylims --> yl
    delete!(plotattributes, :npoints)
    label := auc_label(plotattributes, auc(C, args...), args...)
    delete!(plotattributes, :aucshow)
    return points
end
@recipe function f(::Type{C}, cs::AbstractArray{<:CMVector}, args...) where {C<:AbstractCurve}
    points = apply.(C, cs)
    xl, yl = _lims(points, plotattributes)
    xlims --> xl
    ylims --> yl
    label := auc_label(plotattributes, auc_trapezoidal(points), args...)
    delete!(plotattributes, :aucshow)
    return points
end

function auc_label(plotattributes, auc_score, args...)
    user_label = get(plotattributes, :label, "AUTO")
    if get(plotattributes, :aucshow, false)
        auc_label = string.("auc: ", round.(100 * auc_score', digits = 2), "%")
        if user_label != "AUTO"
            return string.(user_label, " (", auc_label, ")")
        else
            return auc_label
        end
    else
        user_label
    end
end

function _lims((x,y)::Tuple, plotattributes)
    xscale = get(plotattributes, :xscale, :identity)
    yscale = get(plotattributes, :yscale, :identity)
    xlims = xscale === :identity ? (0, 1.01) : (minimum(x[x .> 0]), 1.01)
    ylims = yscale === :identity ? (0, 1.01) : (minimum(y[y .> 0]), 1.01)
    xlims, ylims
end

function _lims(points::Vector{<:Tuple}, plotattributes)
    limss = [_lims(p, plotattributes) for p in points]
    xlimss, ylimss = zip(limss...) |> collect
    xlimss_low = [x[1] for x in xlimss]
    xlimss_high = [x[2] for x in xlimss]
    ylimss_low = [y[1] for y in ylimss]
    ylimss_high = [y[2] for y in ylimss]
    xlims = (minimum(xlimss_low), maximum(xlimss_high))
    ylims = (minimum(ylimss_low), maximum(ylimss_high))
    xlims, ylims
end


# ROC curve
"""
    $(SIGNATURES) 

Returns false positive rates and true positive rates.
"""
@curve ROCCurve
apply(::Type{ROCCurve}, cms::CMVector) = (false_positive_rate(cms), true_positive_rate(cms))
sampling_function(::Type{ROCCurve}) = threshold_at_fpr
# TODO provide a better estimate of these
# smallest possible FPR
function lowest_metric_value(::Type{ROCCurve}, enc::TwoClassEncoding, targets::AbstractVector)
    1.0 / sum(isnegative.(enc, targets))
end

@userplot ROCPlot

@recipe function f(h::ROCPlot)
    seriestype := :mlcurve
    diagonal   --> true
    legend     --> :bottomright
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "ROC curve"
    xguide     --> "false positive rate"
    yguide     --> "true positive rate"
    xgrid      --> true
    ygrid      --> true
    aucshow    --> true

    (ROCCurve, h.args...)
end


# Precision-Recall curve
"""
    $(SIGNATURES) 

Returns recalls and precisions.
"""
@curve PRCurve
apply(::Type{PRCurve}, cms::CMVector) = (recall(cms), precision(cms))
sampling_function(::Type{PRCurve}) = threshold_at_tpr
# TODO provide a better estimate of these
# smallest possible TPR
function lowest_metric_value(::Type{PRCurve}, enc::TwoClassEncoding, targets::AbstractVector)
    1.0 / sum(ispositive.(enc, targets))
end

@userplot PRPlot

@recipe function f(h::PRPlot)
    seriestype := :mlcurve
    legend     --> :bottomleft
    fillrange  --> 0
    fillalpha  --> 0.15
    title      --> "Precision-Recall curve"
    xguide     --> "recall"
    yguide     --> "precision"
    xgrid      --> true
    ygrid      --> true
    aucshow    --> true

    (PRCurve, h.args...)
end
