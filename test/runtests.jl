using Test, EvalMetrics

include("tests_metrics.jl")
include("tests_thresholds.jl")
include("tests_curves.jl")

@testset "All tests" begin
    @testset "Metrics tests" begin
        test_metrics()
    end

    @testset "Thresholds tests" begin
        test_thresholds()
    end

    @testset "Curves tests" begin
        test_curves()
    end
end