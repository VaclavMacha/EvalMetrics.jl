"""
    $(SIGNATURES)

Returns `n` decision thresholds which correspond to `n` evenly spaced quantiles
of the given vector of scores. If `reduced == true`, then the resulting `n` is
`min(length(scores) + 1, n)`. If `zerorecall == true`, then the largest threshold
will be `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.
"""
function thresholds(scores::RealVector, n::Int = length(scores) + 1; reduced::Bool = true, zerorecall::Bool = true)
    ns = length(scores)
    N = reduced ? min(ns + 1, n) : n
    N -= zerorecall
    if N == ns
        thres = sort(scores)
    else
        thres = quantile(scores, range(0, 1, length = N))
    end
    zerorecall && push!(thres, maximum(scores) + eps())

    return thres
end


function threshold_at_rate(enc::TwoClassEncoding, scores::RealVector, rates::RealVector; rev::Bool = true)

    all(0 .<= rates .<= 1) || throw(ArgumentError("input rates out of [0, 1]."))
    issorted(rates) || throw(ArgumentError("input rates must be sorted."))

    n_rates = length(rates)
    n_scores = length(scores)
    print_warn = falses(n_rates)

    # case rate == 1
    if rev
        thresh = fill(scores[end] + eps(scores[end]), n_rates)
        thresh[rates .== 1] .= scores[end]
    else
        thresh = fill(scores[end], n_rates)
    end

    # case rate != 1
    rate_last = 0
    t_last = scores[1]
    j = 1

    for (i, score) in enumerate(scores)
        t_last == score && continue

        # compute current rate
        rate = (i-1)/n_scores 

        for rate_i in rates[j:end]
            rate <= rate_i && break
            rate_last == 0 && rate_i != 0 && (print_warn[j] = true)

            thresh[j] = rev ? t_last + eps(t_last) : t_last
            j += 1
            j > n_rates && (return thresh, print_warn)
        end

        # update last rate and threshold
        rate_last = rate
        t_last = score
    end
    return thresh, print_warn
end


"""
    $(SIGNATURES)

Returns a decision threshold at a given true positive rate `tpr ∈ [0, 1]`.
"""
threshold_at_tpr(targets::AbstractVector, scores::RealVector, tpr) = 
    threshold_at_tpr(current_encoding(), targets, scores, tpr)


threshold_at_tpr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, tpr::Real) = 
    threshold_at_tpr(enc, targets, scores, [tpr])[1]


function threshold_at_tpr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, tpr::RealVector)

    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    scores_pos = sort(scores[ispositive.(enc, targets)]; rev = false)
    rates = round.(1 .- reverse(tpr); digits = 14)
    ts, print_warn = threshold_at_rate(enc, scores_pos, rates; rev = false)

    if any(print_warn)
        rates = tpr[reverse(print_warn)]
        @warn "The closest higher feasible true positive rate to some of the required values ($(join(rates, ", "))) is 1.0!"
    end
    return reverse(ts)
end


"""
    $(SIGNATURES)

Returns a decision threshold at a given true negative rate `fpr ∈ [0, 1]`.
"""
threshold_at_tnr(targets::AbstractVector, scores::RealVector, tnr) = 
    threshold_at_tnr(current_encoding(), targets, scores, tnr)


threshold_at_tnr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, tnr::Real) = 
    threshold_at_tnr(enc, targets, scores, [tnr])[1]


function threshold_at_tnr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, tnr::RealVector)

    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    scores_neg = sort(scores[isnegative.(enc, targets)]; rev = true)
    rates = round.(1 .- reverse(tnr); digits = 14)
    ts, print_warn = threshold_at_rate(enc, scores_neg, rates; rev = true)

    if any(print_warn)
        rates = tnr[reverse(print_warn)]
        @warn "The closest higher feasible true negative rate to some of the required values ($(join(rates, ", "))) is 1.0!"
    end
    return reverse(ts)
end


"""
    $(SIGNATURES)

Returns a decision threshold at a given false positive rate `fpr ∈ [0, 1]`.
"""
threshold_at_fpr(targets::AbstractVector, scores::RealVector, fpr) = 
    threshold_at_fpr(current_encoding(), targets, scores, fpr)


threshold_at_fpr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, fpr::Real) = 
    threshold_at_fpr(enc, targets, scores, [fpr])[1]


function threshold_at_fpr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, fpr::RealVector)

    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    scores_neg = sort(scores[isnegative.(enc, targets)]; rev = true)
    ts, print_warn = threshold_at_rate(enc, scores_neg, fpr; rev = true)

    if any(print_warn)
        rates = fpr[print_warn]
        @warn "The closest lower feasible false positive rate to some of the required values ($(join(rates, ", "))) is 0.0!"
    end
    return ts
end


"""
    $(SIGNATURES)

Returns a decision threshold at a given false negative rate `fnr ∈ [0, 1]`.
"""
threshold_at_fnr(targets::AbstractVector, scores::RealVector, fnr) = 
    threshold_at_fnr(current_encoding(), targets, scores, fnr)


threshold_at_fnr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, fnr::Real) = 
    threshold_at_fnr(enc, targets, scores, [fnr])[1]


function threshold_at_fnr(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, fnr::RealVector)

    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    scores_pos = sort(scores[ispositive.(enc, targets)]; rev = false)
    ts, print_warn = threshold_at_rate(enc, scores_pos, fnr; rev = false)

    if any(print_warn)
        rates = fnr[print_warn]
        @warn "The closest lower feasible false negative rate to some of the required values ($(join(rates, ", "))) is 0.0!"
    end
    return ts
end


"""
    $(SIGNATURES)

Returns a decision threshold at `k` most anomalous samples if `rev == true` and
a decision threshold at `k` least anomalous samples otherwise.
"""
function threshold_at_k(scores::RealVector, k::Int; rev::Bool = true)
    n = length(scores)
    n >= k || throw(ArgumentError("Argument `k` must be smaller or equal to `length(targets) = $n`"))
    return partialsort(scores, k, rev = rev)
end
