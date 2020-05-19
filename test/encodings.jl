using EvalMetrics.Encodings

@testset "ispositive" begin
    @test ispositive(OneZero(), 1)
    @test ispositive(OneTwo(), 1)
    @test ispositive(OneMinusOne(), 1)
    @test ispositive(OneVsRest(1, [2,3]), 1)
    @test ispositive(RestVsOne([1,2,3], 4), 1)
    @test ispositive(RestVsOne([1,2,3], 4), 2)
    @test ispositive(RestVsOne([1,2,3], 4), 3)
end

@testset "ispositive broadcasting" begin
    @test all(ispositive.(OneZero(), [1]))
    @test all(ispositive.(OneTwo(), [1]))
    @test all(ispositive.(OneMinusOne(), [1]))
    @test all(ispositive.(OneVsRest(1, [2,3]), [1]))
    @test all(ispositive.(RestVsOne([1,2,3], 4), [1]))
    @test all(ispositive.(RestVsOne([1,2,3], 4), [2]))
    @test all(ispositive.(RestVsOne([1,2,3], 4), [3]))
end

@testset "isnegative" begin
    @test isnegative(OneZero(), 0)
    @test isnegative(OneTwo(), 2)
    @test isnegative(OneMinusOne(), -1)
    @test isnegative(OneVsRest(1, [2,3]), 2)
    @test isnegative(OneVsRest(1, [2,3]), 3)
    @test isnegative(RestVsOne([1,2,3], 4), 4)
end

@testset "isnegative broadcasting" begin
    @test all(isnegative.(OneZero(), [0]))
    @test all(isnegative.(OneTwo(), [2]))
    @test all(isnegative.(OneMinusOne(), [-1]))
    @test all(isnegative.(OneVsRest(1, [2,3]), [2]))
    @test all(isnegative.(OneVsRest(1, [2,3]), [3]))
    @test all(isnegative.(RestVsOne([1,2,3], 4), [4]))
end

@testset "check_encoding" begin
    @test check_encoding(OneZero(), rand([0,1], 1000))
    @test check_encoding(OneTwo(), rand([1,2], 1000))
    @test check_encoding(OneMinusOne(), rand([1,-1], 1000))
    @test check_encoding(OneVsRest(1, [2,3]), rand([1,2,3], 1000))
    @test check_encoding(RestVsOne([1,2,3], 4), rand([1,2,3,4], 1000))
end