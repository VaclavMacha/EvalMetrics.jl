abstract type AbstractCounts end

struct Counts{T<:Real} <: AbstractCounts
    p::T    # positive in target
    n::T    # negative in target
    tp::T   # correct positive prediction
    tn::T   # correct negative prediction
    fp::T   # (incorrect) positive prediction when target is negative
    fn::T   # (incorrect) negative prediction when target is positive
end


function show(io::IO, x::Counts)
    println(io, "$(typeof(x))")
    println(io, "  p  = $(x.p)")
    println(io, "  n  = $(x.n)")
    println(io, "  tp = $(x.tp)")
    println(io, "  tn = $(x.tn)")
    println(io, "  fp = $(x.fp)")
    println(io, "  fn = $(x.fn)")
end


_ispos(x::Bool) = x
_ispos(x::Real) = x > 0

_predict(x::Real, t::Real) = x >= t

function counts(target::IntegerVector, predict::RealVector)

    length(predict) == length(target) || throw(DimensionMismatch("Inconsistent lengths."))

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for (target_i, predict_i) in zip(target, predict)
        if _ispos(target_i)
            p += 1
            if _ispos(predict_i)
                tp += 1
            else
                fn += 1
            end
        else 
            n += 1
            if _ispos(predict_i)
                fp += 1
            else
                tn += 1
            end
        end
    end
    return Counts{Int}(p, n, tp, tn, fp, fn)
end


function counts(target::IntegerVector, scores::RealVector, threshold::Real)

    length(scores) == length(target) || throw(DimensionMismatch("Inconsistent lengths."))

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for (target_i, scores_i) in zip(target, scores)
        predict_i = _predict(scores_i, threshold)
        if _ispos(target_i)
            p += 1
            if predict_i == target_i
                tp += 1
            else
                fn += 1
            end
        else 
            n += 1
            if _ispos(predict_i)
                fp += 1
            else
                tn += 1
            end
        end
    end
    return Counts{Int}(p, n, tp, tn, fp, fn)
end


# find_thresbin
#
#  x < threshold[1] --> 1
#  threshold[i] <= x < threshold[i+1] --> i+1
#  x >= threshold[n] --> n+1
#
function find_thresbin(x::Real, thresholds::RealVector)

    x <  thresholds[1] && return 1
    n = length(thresholds)
    x >= thresholds[n] && return n + 1

    l, r = 1, n
    while l + 1 < r
        m = (l + r) >> 1 # middle point
        if x < thresholds[m]
            r = m
        else
            l = m
        end
    end
    return r
end


function counts(target::IntegerVector, scores::RealVector, thresholds::RealVector)
    issorted(thresholds) || error("thresholds must be sorted.")
    length(scores) == length(target) || throw(DimensionMismatch("Inconsistent lengths."))

    nt     = length(thresholds)
    bins_p = zeros(Int, nt + 1)
    bins_n = zeros(Int, nt + 1)
    c      = Array{Counts{Int}}(undef, nt)
    p, n   = 0, 0
    fn, tn = 0, 0

    # scan scores and classify them into bins
    for (target_i, scores_i) in zip(target, scores)
        k = find_thresbin(scores_i, thresholds)
        if _ispos(target_i)
            bins_p[k] += 1
            p += 1
        else
            bins_n[k] += 1
            n += 1
        end
    end

    # produce results
    @inbounds for k = 1:nt
        fn  += bins_p[k]
        tn  += bins_n[k]
        tp   = p - fn
        fp   = n - tn
        c[k] = Counts{Int}(p, n, tp, tn, fp, fn)
    end
    return c
end
