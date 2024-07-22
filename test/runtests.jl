using Test, EvalMetrics, Random
using EvalMetrics.Encodings

@testset "All tests" begin
    include("encodings.jl")
    include("utilities.jl")
    include("confusion_matrix.jl")
    include("metrics.jl")
    include("thresholds.jl")
    include("curves.jl")
end