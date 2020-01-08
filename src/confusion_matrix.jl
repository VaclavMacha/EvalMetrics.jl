struct Counts{T<:Real}
    p::T    # positive in target
    n::T    # negative in target
    tp::T   # correct positive prediction
    tn::T   # correct negative prediction
    fp::T   # (incorrect) positive prediction when target is negative
    fn::T   # (incorrect) negative prediction when target is positive
end

show(io::IO, x::Counts) = 
    print(io, "$(typeof(x))$((p = x.p, n = x.n, tp = x.tp, tn = x.tn, fp = x.fp, fn = x.fn))")


_ispos(x::Bool) = x
_ispos(x::Real) = x > 0


_predict(x::Real, t::Real) = x >= t

const CountsVector{T<:Counts} = AbstractArray{T,1}

"""
    counts(target::IntegerVector, predict::RealVector)

For the given prediction `predict` of the true labels `target` computes components
of the binary classification confusion matrix.
"""
function counts(target::IntegerVector, predict::RealVector)

    if length(predict) != length(target)
        throw(DimensionMismatch("Inconsistent lengths of `target` and `predict`."))
    end

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


"""
    counts(target::IntegerVector, scores::RealVector, thres::Real)

For the given prediction `scores .>= thres` of the true labels `target` computes components
of the binary classification confusion matrix.    
"""
function counts(target::IntegerVector, scores::RealVector, thres::Real)

    if length(scores) != length(target)
        throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    end

    p, n, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0

    for (target_i, scores_i) in zip(target, scores)
        predict_i = _predict(scores_i, thres)
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
#  x < thres[1] --> 1
#  thres[i] <= x < thres[i+1] --> i+1
#  x >= thres[n] --> n+1
#
function find_thresbin(x::Real, thres::RealVector)

    x <  thres[1] && return 1
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


"""
    counts(target::IntegerVector, scores::RealVector, thres::RealVector)

For each threshold from `thres` computes components of the binary classification confusion matrix.   
"""
function counts(target::IntegerVector, scores::RealVector, thres::RealVector)

    if !issorted(thres)
        throw(ArgumentError("Thresholds must be sorted."))
    end
    if length(scores) != length(target)
        throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    end

    nt     = length(thres)
    bins_p = zeros(Int, nt + 1)
    bins_n = zeros(Int, nt + 1)
    c      = Array{Counts{Int}}(undef, nt)
    p, n   = 0, 0
    fn, tn = 0, 0

    # scan scores and classify them into bins
    for (target_i, scores_i) in zip(target, scores)
        k = find_thresbin(scores_i, thres)
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
