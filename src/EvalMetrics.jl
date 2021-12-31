module EvalMetrics

using LinearAlgebra
using PrettyTables
using Statistics

import Base: show, precision

include("confusion_matrix.jl")
include("metrics.jl")

export AbstractConfusionMatrix, BinaryConfusionMatrix, ConfusionMatrix, confusion

end
