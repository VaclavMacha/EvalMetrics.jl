module Encodings

export AbstractEncoding, MultiClassEncoding, TwoClassEncoding,
       OneZero, OneMinusOne, OneTwo, OneVsRest, RestVsOne,
       check_encoding, ispositive, isnegative


abstract type AbstractEncoding{T}; end
abstract type MultiClassEncoding{T} <: AbstractEncoding{T}; end 
abstract type TwoClassEncoding{T} <: AbstractEncoding{T}; end 

function Base.show(io::IO, ::MIME"text/plain", enc::T) where {T <: TwoClassEncoding}
    print(io, "$T: \n   positive class: $(enc.positives) \n   negative class: $(enc.negatives)")
end


include("twoclassencodings.jl")


const CURRENT_ENCODING = Ref{AbstractEncoding}(OneZero())

current_encoding() = CURRENT_ENCODING[]
set_encoding(enc::AbstractEncoding) = CURRENT_ENCODING[] = enc
reset_encoding() = CURRENT_ENCODING[] = OneZero()

end