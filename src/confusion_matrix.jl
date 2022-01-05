using Base: ImmutableDict

abstract type AbstractConfusionMatrix{I<:Integer} <: AbstractMatrix{I} end

Base.size(C::AbstractConfusionMatrix) = size(C.data)
Base.getindex(C::AbstractConfusionMatrix, I::Vararg{Int,2}) = getindex(C.data, I...)
Base.setindex!(C::AbstractConfusionMatrix, v, I::Vararg{Int,2}) = setindex!(C.data, v, I...)

function Base.show(io::IO, ::MIME"text/plain", C::AbstractConfusionMatrix)
    N = size(C.data, 2)
    cls = string.(C.classes)
    return pretty_table(
        io,
        Any[[cls...] C.data];
        title = summary(C),
        linebreaks = true,
        header = (
            ["Predictions ŷ →" cls...],
            ["Targets y ↓" repeat([""], N)...],
        ),
        vlines = [0, 1, N + 1],
        alignment = :c,
        crop = :both,
        vcrop_mode = :middle,
        display_size = displaysize(io),
        crop_num_lines_at_beginning = 2,
        header_crayon = Crayon(bold = true, foreground = :blue),
        subheader_crayon = Crayon(bold = true, foreground = :white),
        highlighters = (
            Highlighter(
                f = (data, i, j) -> j == 1,
                crayon = Crayon(bold = true, foreground = :white)
            ),
            Highlighter(
                f = (data, i, j) -> (i + 1) == j,
                crayon = Crayon(bold = true, foreground = :green)
            ),
        ),
        newline_at_end = false
    )
end

function Base.:(+)(C1::AbstractConfusionMatrix, C2::AbstractConfusionMatrix)
    cls1 = C1.classes
    cls2 = C2.classes

    if cls1 == cls2
        C = zero(C1)
        C.data .+= C1.data .+ C2.data
        return C
    elseif issubset(cls1, cls2) && issubset(cls2, cls1)
        prm = [findfirst(isequal(i), cls2) for i in cls1]
        C = zero(C1)
        C.data .+= C1.data .+ C2.data[prm, prm]
        return C
    else
        error("Confusion matrices do not have same classes.")
    end
end

"""
    ConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}

Represenation of a `NxN` confusion matrix with classes of type `T` and elements of type `I`. Rows represent target (or ground truth) and columns represent predictions.

For more information see [`confusion`](@ref) function.

# Fields
- `classes::NTuple{N,T}`: tuple of classes
- `data::Matrix{I}`: confusion matrix
"""
struct ConfusionMatrix{N,T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{N,T}
    data::Matrix{I}

    function ConfusionMatrix(classes::NTuple{N,T}, data::Matrix{I}) where {N,T,I<:Integer}
        if size(data) != (N, N)
            throw(DimensionMismatch("Number of classes must match number of rows and columns of confusion matrix: got length(classes) = $(N) and size(data) = $(size(data) )"))
        end
        return new{N,T,I}(classes, data)
    end

    ConfusionMatrix(classes, data) = ConfusionMatrix((classes...,), Matrix(data))
end

function ConfusionMatrix(classes)
    N = length(classes)
    return ConfusionMatrix(classes, zeros(Int, N, N))
end

Base.zero(C::ConfusionMatrix) = ConfusionMatrix(C.classes)

"""
    BinaryConfusionMatrix{T,I<:Integer} <: AbstractConfusionMatrix{I}

Special representation of `2x2` confusion matrix with classes of type `T` and elements of type `I`. Rows represent target (or ground truth) and columns represent predictions. 

For more information see [`confusion`](@ref) function.

# Fields
- `classes::NTuple{2,T}`: tuple of classes
- `data::Matrix{I}`: confusion matrix
"""
struct BinaryConfusionMatrix{T,I<:Integer} <: AbstractConfusionMatrix{I}
    classes::NTuple{2,T}
    data::Matrix{I}

    function BinaryConfusionMatrix(
        classes::NTuple{2,T},
        data::Matrix{I}
    ) where {T,I<:Integer}

        if size(data) != (2, 2)
            throw(DimensionMismatch("Number of classes must match confusion matrix size: got length(classes) = $(2) and size(data) = $(size(data) )"))
        end
        return new{T,I}(classes, data)
    end

    function BinaryConfusionMatrix(classes, data)
        return BinaryConfusionMatrix((classes...,), Matrix(data))
    end
end

function BinaryConfusionMatrix(classes)
    N = length(classes)
    return BinaryConfusionMatrix(classes, zeros(Int, N, N))
end

Base.zero(C::BinaryConfusionMatrix) = BinaryConfusionMatrix(C.classes)

# Constructors
labelmap(C::AbstractConfusionMatrix) = labelmap(C.classes)
labelmap(classes) = ImmutableDict(Pair.(classes, eachindex(classes))...)

"""
    confusion(y, ŷ)

General constructor for [`ConfusionMatrix`](@ref) and [`BinaryConfusionMatrix`](@ref).

# keyword arguments
- `classes`: vector-like object of classes. Default value is `sort(unique(y))`.
- `binary::Bool`: if `true` the function returns BinaryConfusionMatrix in case of a binary problem. Default value is `true`.
- `target_transform`: function that is aplied element-wisely to transform targets `y` to correspond to given classes, i.e. it must hold `target_transform(y[i]) ∈ classes` for all indices. Default transform is `identity`.
- `predict_transform`: function that is aplied element-wisely to transform predictions `ŷ` to correspond to given classes, i.e. it must hold `predict_transform(y[i]) ∈ classes` for all indices. Default transform is `identity`.

# Examples

The function can be used to create confusion matrix directly from targets and predictions

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> confusion(y, ŷ)
2×2 BinaryConfusionMatrix{Int64, Int64}
┌─────────────────┬──────┐
│ Predictions ŷ → │ 0  1 │
│   Targets y ↓   │      │
├─────────────────┼──────┤
│        0        │ 2  2 │
│        1        │ 3  3 │
└─────────────────┴──────┘
```
or for example to create confusion matrix directly from targets, scores and decision threshold

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> s = [0.4, 0.7, 0.2, 0.6, 0.8, 0.4, 0.5, 0.3, 0.9, 0.7];

julia> t = 0.6;

julia> confusion(y, ŷ; predict_transform = s -> s >= t)
2×2 BinaryConfusionMatrix{Int64, Int64}
┌─────────────────┬──────┐
│ Predictions ŷ → │ 0  1 │
│   Targets y ↓   │      │
├─────────────────┼──────┤
│        0        │ 2  2 │
│        1        │ 3  3 │
└─────────────────┴──────┘
```
Similarly, the function can be used to create multi-class confusion matrix
```jldoctest
julia> y = [:a, :b, :b, :c, :b, :a, :c, :a, :b, :c];

julia> ŷ = [:b, :a, :b, :c, :c, :a, :b, :a, :c, :c];

julia> confusion(y, ŷ)
3×3 ConfusionMatrix{3, Symbol, Int64}
┌─────────────────┬─────────┐
│ Predictions ŷ → │ a  b  c │
│   Targets y ↓   │         │
├─────────────────┼─────────┤
│        a        │ 2  1  0 │
│        b        │ 1  1  2 │
│        c        │ 0  1  2 │
└─────────────────┴─────────┘
```
"""
function confusion(
    y,
    ŷ;
    classes = sort(unique(y)),
    binary::Bool = true,
    target_transform = identity,
    predict_transform = identity
)

    if !allunique(classes)
        throw(ArgumentError("Classes must be unique."))
    end
    if length(y) != length(ŷ)
        throw(DimensionMismatch("Inconsistent lengths of y and ŷ."))
    end
    if binary && length(classes) == 2
        C = BinaryConfusionMatrix(classes)
    else
        C = ConfusionMatrix(classes)
    end
    mapping = labelmap(C)

    for (y_i, ŷ_i) in zip(y, ŷ)
        i = mapping[target_transform(y_i)]
        j = mapping[predict_transform(ŷ_i)]
        @inbounds C[i, j] += 1
    end
    return C
end

"""
    confusion(y::AbstractMatrix, ŷ::AbstractMatrix)

Constructor for [`ConfusionMatrix`](@ref) and [`BinaryConfusionMatrix`](@ref) for cases where both targets and predictions are in a form of matrix, for example when the one-hot encoding is used.

# Examples

```jldoctest
julia> y = [
           1 0 0 0 0 1 0 1 0 0
           0 1 1 0 1 0 0 0 1 0
           0 0 0 1 0 0 1 0 0 1
       ];

julia> ŷ = [
           0 1 0 0 0 1 0 1 0 0
           1 0 1 0 0 0 1 0 0 0
           0 0 0 1 1 0 0 0 1 1
       ];

julia> confusion(y, ŷ)
3×3 ConfusionMatrix{3, Int64, Int64}
┌─────────────────┬─────────┐
│ Predictions ŷ → │ 1  2  3 │
│   Targets y ↓   │         │
├─────────────────┼─────────┤
│        1        │ 2  1  0 │
│        2        │ 1  1  2 │
│        3        │ 0  1  2 │
└─────────────────┴─────────┘
```
"""
function confusion(
    y::AbstractMatrix,
    ŷ::AbstractMatrix;
    obsdim = 2,
    binary::Bool = true,
    target_transform = argmax,
    predict_transform = argmax,
)

    return confusion(
        eachslice(y; dims = obsdim),
        eachslice(ŷ; dims = obsdim);
        classes = 1:size(y, obsdim == 1 ? 2 : 1),
        binary,
        target_transform,
        predict_transform,
    )
end