module Encodings

export 
    # encodings
    AbstractEncoding,
    MultiClassEncoding,
    TwoClassEncoding,
    OneZero,
    OneMinusOne,
    OneTwo,
    OneVsOne,
    OneVsRest,
    RestVsOne,
    
    # utility functions
    check_encoding,
    ispositive,
    isnegative,
    current_encoding,
    set_encoding,
    reset_encoding,
    recode,
    classify


abstract type AbstractEncoding{T}; end
abstract type MultiClassEncoding{T} <: AbstractEncoding{T}; end
abstract type TwoClassEncoding{T} <: AbstractEncoding{T}; end

Base.show(io::IO, ::MIME"text/plain", enc::T) where {T <: TwoClassEncoding} =
    print(io, "$T: \n   positive class: $(enc.positives) \n   negative class: $(enc.negatives)")


include("twoclassencodings.jl")
include("utilities.jl")

const CURRENT_ENCODING = Ref{AbstractEncoding}(OneZero())

current_encoding() = CURRENT_ENCODING[]
set_encoding(enc::AbstractEncoding) = CURRENT_ENCODING[] = enc
reset_encoding() = CURRENT_ENCODING[] = OneZero()

end