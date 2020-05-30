check_encoding(enc::TwoClassEncoding, targets) =
    all(val == enc.positives || val == enc.negatives for val in unique(targets))


ispositive(enc::TwoClassEncoding, val) = _ispositive(enc, val)
_ispositive(enc::TwoClassEncoding, val) = isone(val)

Broadcast.broadcasted(::typeof(ispositive), enc, val) =
    broadcast(_ispositive, Ref(enc), val)


isnegative(enc::TwoClassEncoding, val) = _isnegative(enc, val)
_isnegative(enc::TwoClassEncoding, val) = !_ispositive(enc, val)

Broadcast.broadcasted(::typeof(isnegative), enc, val) =
    broadcast(_isnegative, Ref(enc), val)


positive_label(enc::TwoClassEncoding) = enc.positives
negative_label(enc::TwoClassEncoding) = enc.negatives


"""
    OneZero{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `zero(T)` the negative class.
"""
struct OneZero{T<:Number} <: TwoClassEncoding{T}
    positives::T
    negatives::T    

    OneZero(::Type{T} = Float64) where {T<:Number} =
        new{T}(one(T), zero(T))    
end


"""
    OneMinusOne{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `-one(T)` the negative class.
"""
struct OneMinusOne{T<:Number} <: TwoClassEncoding{T}
    positives::T
    negatives::T    

    OneMinusOne(::Type{T} = Float64) where {T<:Number} =
        new{T}(one(T), -one(T))    
end


"""
    OneTwo{T<:Number} <: TwoClassEncoding{T}

Two class label encoding in which `one(T)` represents the positive class,
and `2*one(T)` the negative class.
"""
struct OneTwo{T<:Number} <: TwoClassEncoding{T}
    positives::T
    negatives::T    

    OneTwo(::Type{T} = Float64) where {T<:Number} =
        new{T}(one(T), 2*one(T))    
end


"""
    OneVsOne{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct OneVsOne{T} <: TwoClassEncoding{T}
    positives::T
    negatives::T

    OneVsOne(pos::P, neg::N, ::Type{T} = default_type(P, N)) where {P, N, T} = 
        new{T}(T(pos), T(neg))
end

_ispositive(enc::OneVsOne, val) = val == enc.positives


"""
    OneVsRest{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct OneVsRest{T} <: TwoClassEncoding{T}
    positives::T
    negatives::AbstractVector{T}

    OneVsRest(pos::P, neg::AbstractVector{N}, ::Type{T} = default_type(P, N)) where {P, N, T} = 
        new{T}(T(pos), T.(neg))
end


check_encoding(enc::OneVsRest, targets) =
    all(val == enc.positives || (val in enc.negatives) for val in unique(targets))

_ispositive(enc::OneVsRest, val) = val == enc.positives
negative_label(enc::OneVsRest) = enc.negatives[1]


"""
    RestVsOne{T} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct RestVsOne{T} <: TwoClassEncoding{T}
    positives::AbstractVector{T}
    negatives::T    

    RestVsOne(pos::AbstractVector{P}, neg::N, ::Type{T} = default_type(P, N)) where {P, N, T} = 
        new{T}(T.(pos), T(neg))
end


check_encoding(enc::RestVsOne, targets) =
    all(val in enc.positives || val == enc.negatives for val in unique(targets))

_ispositive(enc::RestVsOne, val) = val != enc.negatives
positive_label(enc::RestVsOne) = enc.positives[1]