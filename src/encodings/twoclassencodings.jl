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
    OneVsRest{T<:Number} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct OneVsRest{T<:Number} <: TwoClassEncoding{T}
    positives::T
    negatives::AbstractVector{T}    

    OneVsRest(pos, neg, ::Type{T} = Float64) where {T} =
        new{T}(T(pos), T.(neg))
end


"""
    RestVsOne{T<:Number} <: TwoClassEncoding{T}

Two class label encoding ...
"""
struct RestVsOne{T<:Number} <: TwoClassEncoding{T}
    positives::AbstractVector{T}
    negatives::T    

    RestVsOne(pos, neg, ::Type{T} = Float64) where {T} =
        new{T}(T.(pos), T(neg))
end


"""
    check_encoding(enc::AbstractEncoding, targets)

...
"""
check_encoding(enc::TwoClassEncoding, targets) =
    all(val == enc.positives || val == enc.negatives for val in unique(targets))

check_encoding(enc::OneVsRest, targets) =
    all(val == enc.positives || (val in enc.negatives) for val in unique(targets))

check_encoding(enc::RestVsOne, targets) =
    all(val in enc.positives || val == enc.negatives for val in unique(targets))


ispositive(enc::TwoClassEncoding, val) = _ispositive(enc, val)
_ispositive(enc::TwoClassEncoding, val) = isone(val)
_ispositive(enc::OneVsRest, val) = val == enc.positives
_ispositive(enc::RestVsOne, val) = val != enc.negatives

Broadcast.broadcasted(::typeof(ispositive), enc, val) =
    broadcast(_ispositive, Ref(enc), val)

isnegative(enc::TwoClassEncoding, val) = _isnegative(enc, val)
_isnegative(enc::TwoClassEncoding, val) = !_ispositive(enc, val)

Broadcast.broadcasted(::typeof(isnegative), enc, val) =
    broadcast(_isnegative, Ref(enc), val)




