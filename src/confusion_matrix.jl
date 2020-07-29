struct ConfusionMatrix{T<:Real}
    p::T    # positive in target
    n::T    # negative in target
    tp::T   # correct positive prediction
    tn::T   # correct negative prediction
    fp::T   # (incorrect) positive prediction when target is negative
    fn::T   # (incorrect) negative prediction when target is positive
end


function Base.:(+)(a::ConfusionMatrix{T}, b::ConfusionMatrix{S}) where {T, S}
    ConfusionMatrix{promote_type(T,S)}(
        a.p + b.p,
        a.n + b.n,
        a.tp + b.tp,
        a.tn + b.tn,
        a.fp + b.fp,
        a.fn + b.fn
    )
end


"""
    ConfusionMatrix(targets::AbstractVector, predicts::AbstractVector)
    ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, predicts::AbstractVector)

For the given prediction `predicts` of the true labels `targets` computes the binary confusion matrix.
"""
ConfusionMatrix(targets::AbstractVector, predicts::AbstractVector) =
    ConfusionMatrix(current_encoding(), targets, predicts)


function ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, predicts::AbstractVector)
    length(targets) == length(predicts) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `predicts`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))
    check_encoding(enc, predicts) || throw(ArgumentError("`predicts` vector uses incorrect label encoding."))

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for i in eachindex(targets)
        @inbounds tar  = ispositive(enc, targets[i])
        @inbounds pred = ispositive(enc, predicts[i])

        if tar
            p += 1
            pred ? (tp += 1) : (fn += 1)
        else
            n += 1
            pred ? (fp += 1) : (tn += 1)
        end
    end
    return ConfusionMatrix{Int}(p, n, tp, tn, fp, fn)
end


"""
    ConfusionMatrix(targets::AbstractVector, scores::RealVector, thres::Real)
    ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, thres::Real)

For the given prediction `scores .>= thres` of the true labels `targets` computes
the binary confusion matrix.
"""
ConfusionMatrix(targets::AbstractVector, scores::RealVector, thres::Real) =
    ConfusionMatrix(current_encoding(), targets, scores, thres)


function ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, thres::Real)
    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for i in eachindex(targets)
        @inbounds tar  = ispositive(enc, targets[i])
        @inbounds pred = ispositive(enc, classify(enc, scores[i], thres))

        if tar
            p += 1
            pred ? (tp += 1) : (fn += 1)
        else
            n += 1
            pred ? (fp += 1) : (tn += 1)
        end
    end
    return ConfusionMatrix{Int}(p, n, tp, tn, fp, fn)
end


"""
    ConfusionMatrix(targets::AbstractVector, scores::RealVector, thres::RealVector)
    ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, thres_in::RealVector)

For each threshold from `thres` computes the binary classification confusion matrix.
"""
ConfusionMatrix(targets::AbstractVector, scores::RealVector, thres::RealVector) =
    ConfusionMatrix(current_encoding(), targets, scores, thres)


function ConfusionMatrix(enc::TwoClassEncoding, targets::AbstractVector, scores::RealVector, thres_in::RealVector)
    flag_rev = false
    thres = thres_in

    if issorted(thres)
        flag_rev = false
    elseif issorted(thres; rev = true)
        thres = reverse(thres_in)
        flag_rev = true
    else
        throw(ArgumentError("Thresholds must be sorted."))
    end

    length(targets) == length(scores) || throw(DimensionMismatch("Inconsistent lengths of `targets` and `scores`."))
    check_encoding(enc, targets) || throw(ArgumentError("`targets` vector uses incorrect label encoding."))

    # scan scores and classify them into bins
    nt = length(thres)
    bins_p = zeros(Int, nt + 1)
    bins_n = zeros(Int, nt + 1)
    p, n = 0, 0

    for i in eachindex(targets)
        @inbounds tar = ispositive(enc, targets[i])
        @inbounds scr = scores[i]

        k = find_threshold_bins(scr, thres)

        if tar
            bins_p[k] += 1
            p += 1
        else
            bins_n[k] += 1
            n += 1
        end
    end

    # produce results
    c = Array{ConfusionMatrix{Int}}(undef, nt)
    fn, tn = 0, 0

    @inbounds for k = 1:nt
        fn += bins_p[k]
        tn += bins_n[k]
        tp = p - fn
        fp = n - tn
        c[k] = ConfusionMatrix{Int}(p, n, tp, tn, fp, fn)
    end
    return flag_rev ? reverse(c) : c
end


"""
    find_threshold_bins(x::Real, thres::RealVector)

find_threshold_bins:
    x < thres[1] --> 1
    thres[i] <= x < thres[i+1] --> i+1
    x >= thres[n] --> n+1
"""
function find_threshold_bins(x::Real, thres::RealVector)
    x < thres[1] && return 1
    n = length(thres)
    x >= thres[n] && return n + 1

    l, r = 1, n
    while l + 1 < r
        m = (l + r) >> 1 # middle point
        if x < thres[m]
            r = m
        else
            l = m
        end
    end
    return r
end
