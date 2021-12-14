module EvalMetrics

using LinearAlgebra
using Statistics

import Base: show, precision

include("confusion_matrix.jl")
include("metrics.jl")

export AbstractConfusionMatrix, BinaryConfusionMatrix, ConfusionMatrix

end
