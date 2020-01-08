"""
    thresholds(scores::RealVector, n::Int [; reduced::Bool = true, zerorecall::Bool = true])

Returns `n` decision thresholds which correspond to `n` evenly spaced quantiles
of the given vector of scores. If `reduced == true`, then the resulting `n` is
`min(length(scores) + 1, n)`. If `zerorecall == true`, then the largest threshold
will be `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.
"""
function thresholds(scores, n::Int = length(scores) + 1; reduced::Bool = true, zerorecall::Bool = true)
    ns = length(scores)
    N  = reduced    ? min(ns + 1, n) : n
    N  = zerorecall ? N - 1 : N
    if N == ns
        thres = sort(scores)
    else
        thres = quantile(scores, range(0, 1, length = N))
    end
    return zerorecall ? vcat(thres, maximum(scores)*(1 + eps())) : thres
end


"""
    threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)

Returns a decision threshold at a given false positive rate `fpr ∈ [0, 1]`.
"""
function threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)
    n     = length(target)
    n_pos = sum(_ispos.(target))
    n_neg = n - n_pos 

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
        if !_ispos(target[i])
            l += 1
            if (l - 1)/n_neg <= fpr && l/n_neg > fpr
                break
            end
        end
    end
    scores[prm[k]] + eps()
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
    n_pos = sum(_ispos.(target))

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
        if _ispos(target[i])
            l += 1
            if (l - 1)/n_pos <= fnr && l/n_pos > fnr
                break
            end
        end
    end
    scores[prm[k]]
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