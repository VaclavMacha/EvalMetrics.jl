"""
    auc(x::RealVector, y::RealVector)

Computes the area under curve `(x,y)` using trapezoidal rule.
"""
function auc(x::RealVector, y::RealVector)
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

auprc(counts::Vector{Counts{T}}) where T = auc(recall.(counts), precision.(counts))
auprc(target::LabelVector, scores::RealVector; classes::Tuple = (0, 1)) =
    auprc(target, scores, thresholds(scores); classes=classes)

function auprc(target::LabelVector, scores::RealVector, thres::RealVector; classes::Tuple=(0, 1))
    n = length(scores)
    n == length(target) || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    ispos = get_ispos(classes)
    0 < sum(ispos.(target)) || throw(ArgumentError("No positive samples present in `target`."))
    auprc(counts(target, scores, thres; classes=classes))
end


auroc(counts::Vector{Counts{T}}) where T = auc(true_positive_rate.(counts), true_negative_rate.(counts))
auroc(target::LabelVector, scores::RealVector; classes::Tuple=(0, 1)) =
    auroc(target, scores, thresholds(scores); classes=classes)

function auroc(target::LabelVector, scores::RealVector, thres::RealVector; classes::Tuple=(0, 1))
    n = length(scores)
    n == length(target) || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    ispos = get_ispos(classes)
    0 < sum(ispos.(target)) < n || throw(ArgumentError("Only one class present in `target`."))
    auroc(counts(target, scores, thres; classes=classes))
end

