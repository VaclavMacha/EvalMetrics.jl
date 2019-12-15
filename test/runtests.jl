using Test, EvalMetrics

include("tests_metrics.jl")

@testset "All tests" begin
    @testset "Metrics tests" begin
        test_metrics()
    end
end