"""
    thresholds(scores::RealVector, n::Int [; reduced::Bool = true, zerorecall::Bool = true])

Returns `n` decision thresholds which correspond to `n` evenly spaced quantiles
of the given vector of scores. If `reduced == true`, then the resulting `n` is
`min(length(scores) + 1, n)`. If `zerorecall == true`, then the largest threshold
will be `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.
"""
function thresholds(scores::RealVector, n::Int = length(scores) + 1; reduced::Bool = true, zerorecall::Bool = true)
    ns = length(scores)
    N  = reduced    ? min(ns + 1, n) : n
    N  = zerorecall ? N - 1 : N
    if N == ns
        thres = sort(scores)
    else
        thres = quantile(scores, range(0, 1, length = N))
    end
    if zerorecall
        s_max = maximum(scores)
        push!(thres, s_max + abs(s_max)*eps())
    end
    return thres
end


"""
    threshold_at_fpr(target::LabelVector, scores::RealVector, fpr::Real)

Returns a decision threshold at a given false positive rate `fpr ∈ [0, 1]`.
"""
threshold_at_fpr(target::LabelVector, scores::RealVector, fpr::Real; kwargs...) = 
    threshold_at_fpr(target, scores, [fpr]; kwargs...)[1]


function threshold_at_fpr(target::LabelVector, scores::RealVector, fpr::RealVector; classes::Tuple = (0, 1))
    ispos = get_ispos(classes)
    n     = length(target)
    n_pos = sum(ispos.(target))
    n_neg = n - n_pos 
    m_max = length(fpr)

    n == length(scores)  || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    all(0 .<= fpr .<= 1) || throw(ArgumentError("input false positive rates out of [0, 1]."))
    issorted(fpr)        || throw(ArgumentError("input false positive rates must be sorted."))


    if issorted(scores, rev = false)
        prm = n:-1:1
    elseif issorted(scores, rev = true)
        prm = 1:n
    else
        prm = sortperm(scores, rev = true)
    end

    thresh = zeros(eltype(scores), size(fpr)...)
    m_start, m_stop = 1, m_max

    for m in 1:m_max
        if fpr[m] == 0
            thresh[m]  = maximum(scores) + eps()
            m_start  += 1
        else
            break
        end
    end

    for m in m_max:-1:1
        if fpr[m] == 1
            thresh[m] = minimum(scores)
            m_stop  -= 1
        else
            break
        end
    end

    if m_stop < m_start
        return thresh
    end


    k, l, m = 0, 0, m_start
    for i in prm
        k += 1
        if !ispos(target[i])
            l += 1
            while l/n_neg > fpr[m]
                thresh[m] = scores[prm[k]] + eps()
                m        += 1
                m > m_stop && break
            end
            m > m_stop && break
        end
    end
    return thresh
end


"""
    threshold_at_tnr(target::LabelVector, scores::RealVector, tnr::Real)

Returns a decision threshold at a given true negative rate `fpr ∈ [0, 1]`.
"""
threshold_at_tnr(target::LabelVector, scores::RealVector, tnr::Real; kwargs...) = 
    threshold_at_fpr(target, scores, 1 - tnr; kwargs...)

threshold_at_tnr(target::LabelVector, scores::RealVector, tnr::RealVector; kwargs...) = 
    reverse(threshold_at_fpr(target, scores, 1 .- reverse(tnr); kwargs...))


"""
    threshold_at_tpr(target::LabelVector, scores::RealVector, tpr::Real)

Returns a decision threshold at a given true positive rate `tpr ∈ [0, 1]`.
"""
threshold_at_tpr(target::LabelVector, scores::RealVector, tpr::Real; kwargs...) = 
    threshold_at_fnr(target, scores, 1 - tpr; kwargs...)

threshold_at_tpr(target::LabelVector, scores::RealVector, tpr::RealVector; kwargs...) = 
    reverse(threshold_at_fnr(target, scores, 1 .- reverse(tpr); kwargs...))


"""
    threshold_at_fnr(target::LabelVector, scores::RealVector, fnr::Real)

Returns a decision threshold at a given false negative rate `fnr ∈ [0, 1]`.
"""
threshold_at_fnr(target::LabelVector, scores::RealVector, fnr::Real; kwargs...) = 
    threshold_at_fnr(target, scores, [fnr]; kwargs...)[1]


function threshold_at_fnr(target::LabelVector, scores::RealVector, fnr::RealVector; classes::Tuple = (0, 1))
    ispos = get_ispos(classes)
    n     = length(target)
    n_pos = sum(ispos.(target))
    m_max = length(fnr)

    n == length(scores)  || throw(DimensionMismatch("Inconsistent lengths of `target` and `scores`."))
    all(0 .<= fnr .<= 1) || throw(ArgumentError("input false negative rates out of [0, 1]."))
    issorted(fnr)        || throw(ArgumentError("input false negative rates must be sorted."))


    if issorted(scores, rev = false)
        prm = 1:n
    elseif issorted(scores, rev = true)
        prm = n:-1:1
    else
        prm = sortperm(scores)
    end

    thresh = zeros(eltype(scores), size(fnr)...)
    m_start, m_stop = 1, m_max
    
    for m in 1:m_max
        if fnr[m] == 0
            thresh[m]  = minimum(scores)
            m_start  += 1
        else
            break
        end
    end

    for m in m_max:-1:1
        if fnr[m] == 1
            thresh[m] = maximum(scores) + eps()
            m_stop  -= 1
        else
            break
        end
    end

    if m_stop < m_start
        return thresh
    end

    k, l, m = 0, 0, m_start
    for i in prm
        k += 1
        if ispos(target[i])
            l += 1
            while l/n_pos > fnr[m]
                thresh[m] = scores[prm[k]]
                m        += 1
                m > m_stop && break
            end
            m > m_stop && break
        end
    end
    return thresh
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