using Base: ImmutableDict

abstract type AbstractConfusionMatrix end

struct ConfusionMatrix{N,T,I<:Integer} <: AbstractMatrix{Int}
    data::Matrix{I}
    mapping::ImmutableDict{T,I}

    function ConfusionMatrix(classes)
        N = length(classes)
        mapping = ImmutableDict([class => i for (i, class) in enumerate(classes)]...)
        return new{N,keytype(mapping),Int}(zeros(Int, N, N), mapping)
    end
end

const BinaryConfusionMatrix{T,I<:Integer} = ConfusionMatrix{2,T,I}

Base.size(C::ConfusionMatrix) = size(C.data)
Base.zero(::ConfusionMatrix{I}) where {I} = ConfusionMatrix(I, size(C, 1))
Base.getindex(C::ConfusionMatrix, I::Vararg{Int,2}) = getindex(C.data, I...)
Base.setindex!(C::ConfusionMatrix, v, I::Vararg{Int,2}) = setindex!(C.data, v, I...)

function fill!(C::ConfusionMatrix, y, ŷ, predict = identity)
    if length(y) != length(ŷ)
        throw(DimensionMismatch("Inconsistent lengths of targets and predictions."))
    end

    for (yi, ŷi) in zip(y, ŷ)
        @inbounds C[C.mapping[yi], C.mapping[predict(ŷi)]] += 1
    end
    return
end

function ConfusionMatrix(y, ŷ; classes = sort(unique(y)))
    C = ConfusionMatrix(classes)
    fill!(C, y, ŷ)
    return C
end

function ConfusionMatrix(y, s, t; classes = sort(unique(y)))
    C = ConfusionMatrix(classes)
    fill!(C, y, s, s -> s >= t)
    return C
end
