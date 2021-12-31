using Base: ImmutableDict

abstract type AbstractConfusionMatrix{I<:Integer} <: AbstractMatrix{I} end

struct ConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{N,T}
    data::Matrix{I}

    function ConfusionMatrix(classes)
        N = length(classes)
        cls = (classes...,)
        return new{N,eltype(cls),Int}(cls, zeros(Int, N, N))
    end
end

struct BinaryConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{N,T}
    data::Matrix{I}

    function BinaryConfusionMatrix(classes)
        if length(classes) != 2
            throw(DimensionMismatch("BinaryConfusionMatrix is defined only for 2D problems: got $(N)D input. Use ConfusionMatrix instead."))
        end
        cls = (classes...,)
        return new{2,eltype(cls),Int}(cls, zeros(Int, 2, 2))
    end
end

Base.size(C::AbstractConfusionMatrix) = size(C.data)
Base.getindex(C::AbstractConfusionMatrix, I::Vararg{Int,2}) = getindex(C.data, I...)
Base.setindex!(C::AbstractConfusionMatrix, v, I::Vararg{Int,2}) = setindex!(C.data, v, I...)

labelmap(C::AbstractConfusionMatrix) = labelmap(C.classes)
labelmap(classes) = ImmutableDict(Pair.(classes, eachindex(classes))...)

function fill!(C::AbstractConfusionMatrix, y, ŷ, predict = identity)
    if length(y) != length(ŷ)
        throw(DimensionMismatch("Inconsistent lengths of targets and predictions."))
    end
    mapping = labelmap(C)

    for (yi, ŷi) in zip(y, ŷ)
        @inbounds C[mapping[yi], mapping[predict(ŷi)]] += 1
    end
    return
end

function confusion(classes; binary::Bool = true)
    N = length(classes)

    return if binary && N == 2
        BinaryConfusionMatrix(classes)
    else
        ConfusionMatrix(classes)
    end
end

function confusion(y, ŷ; binary::Bool = true, classes = sort(unique(y)))
    C = confusion(classes; binary)
    fill!(C, y, ŷ)
    return C
end

function confusion(y, s, t; binary::Bool = true, classes = sort(unique(y)))
    C = confusion(classes; binary)
    fill!(C, y, s, s -> s >= t)
    return C
end
