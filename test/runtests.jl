using Test, EvalMetrics, Random

Random.seed!(42)

include("tests_metrics.jl")
include("tests_thresholds.jl")
include("tests_curves.jl")

@testset "Metrics tests" begin
    test_metrics()
end

@testset "Thresholds tests" begin
    test_thresholds()
end

@testset "Curves tests" begin
    test_auc()
    test_roc_pr()
end
