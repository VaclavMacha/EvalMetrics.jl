default_type(ts::Type...) = default_type(promote_type(ts...))
default_type(::Type{T}) where {T} = T
default_type(::Type{<:Number}) = Float64


recode(enc::AbstractEncoding, enc_new::AbstractEncoding, x) =
    _recode(enc, enc_new, x)

Broadcast.broadcasted(::typeof(recode), enc, enc_new, x) =
    broadcast(_recode, Ref(enc), Ref(enc_new), x)

_recode(enc::TwoClassEncoding, enc_new::TwoClassEncoding, x) =
    ispositive(enc, x) ? positive_label(enc_new) : negative_label(enc_new)


classify(enc::TwoClassEncoding, score, t) =
    _classify(enc, score, t)

Broadcast.broadcasted(::typeof(classify), enc, score, t) =
    broadcast(_classify, Ref(enc), score, t)

_classify(enc::TwoClassEncoding, score, t) =
    score .>= t ? positive_label(enc) : negative_label(enc)