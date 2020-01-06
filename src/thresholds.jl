"""
    thresholds(scores::RealVector, n::Int [; reduced::Bool = true])

Returns `n` decision thresholds which correspond to `n` evenly spaced quantiles of
the given vector of scores. If the keyword argument `reduced == true`,
then the resulting `n` is `min(length(scores), n)`.
"""
function thresholds(scores::RealVector{T}, n::Int = length(scores); reduced::Bool = true) where {T}
    N  = reduced ? min(length(scores), n) : n
    ts = quantile(scores, range(0, 1, length = N))
    ts[1]   -= eps(T)
    ts[end] += eps(T)
    return ts
end


"""
    threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)

Returns a decision threshold at a given false positive rate `fpr ∈ [0, 1]`.
"""
function threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)
    n     = length(target)
    n_neg = n - sum(target) 

    n == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    0 <= fpr <= 1       || throw(ArgumentError("Argument `fpr` must be from interval [0, 1]."))
    fpr >= 1/n_neg      || throw(ArgumentError("No score to estimate `fpr` lower than $(1/n_neg)."))

    fpr == 0 && return maximum(scores) + eps()
    fpr == 1 && return minimum(scores)

    if issorted(scores, rev = true)
        prm = 1:n
    else
        prm = sortperm(scores, rev = true)
    end

    k, l = 0, 0
    for i in prm
        k += 1
        if target[i] == 0
            l += 1
            l/n_neg >= fpr && break
        end
    end

    if l/n_neg == fpr
        return scores[prm[k]]
    else
        interpolate(fpr, scores[prm[k-1]], scores[prm[k]], (l-1)/n_neg, l/n_neg)
    end
end


"""
    threshold_at_tnr(target::IntegerVector, scores::RealVector, tnr::Real)

Returns a decision threshold at a given true negative rate `fpr ∈ [0, 1]`.
"""
threshold_at_tnr(target::IntegerVector, scores::RealVector, tnr::Real) = 
    threshold_at_fpr(target, scores, 1 - tnr)


"""
    threshold_at_tpr(target::IntegerVector, scores::RealVector, tpr::Real)

Returns a decision threshold at a given true positive rate `tpr ∈ [0, 1]`.
"""
threshold_at_tpr(target::IntegerVector, scores::RealVector, tpr::Real) = 
    threshold_at_fnr(target, scores, 1 - tpr)


"""
    threshold_at_fnr(target::IntegerVector, scores::RealVector, fnr::Real)

Returns a decision threshold at a given false negative rate `fnr ∈ [0, 1]`.
"""
function threshold_at_fnr(target::IntegerVector, scores::RealVector, fnr::Real)
    n     = length(target)
    n_pos = sum(target) 

    n == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    0 <= fnr <= 1       || throw(ArgumentError("Argument `fnr` must be from interval [0, 1]."))
    fnr >= 1/n_pos      || throw(ArgumentError("No score to estimate `fnr` lower than $(1/n_pos)."))

    fnr == 0 && return maximum(scores) + eps()
    fnr == 1 && return minimum(scores)

    if issorted(scores)
        prm = 1:n
    else
        prm = sortperm(scores)
    end

    k, l = 0, 0
    for i in prm
        k += 1
        if target[i] == 1
            l += 1
            l/n_pos >= fnr && break
        end
    end
    if l/n_pos == fnr
        return scores[prm[k]]
    else
        interpolate(fnr, scores[prm[k-1]], scores[prm[k]], (l-1)/n_pos, l/n_pos)
    end
end


"""
    threshold_at_k(scores::RealVector, k::Int[; rev::Bool = true])

Returns a decision threshold at `k` most anomalous samples if `rev == true` and
a decision threshold at `k` least anomalous samples otherwise.
"""
function threshold_at_k(scores::RealVector, k::Int; rev::Bool = true)

    length(scores) >= k || throw(ArgumentError("Argument `k` must be smaller or equal to `length(target) = $n`"))

    return partialsort(scores, k, rev = rev)
end