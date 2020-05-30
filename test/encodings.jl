using EvalMetrics.Encodings
import EvalMetrics.Encodings: positive_label, negative_label


@testset "ispositive" begin
    @test ispositive(OneZero(), 1)
    @test ispositive(OneTwo(), 1)
    @test ispositive(OneMinusOne(), 1)
    @test ispositive(OneVsRest(1, [2,3]), 1)
    @test ispositive(RestVsOne([1,2], 3), 1)
    @test ispositive(RestVsOne([1,2], 3), 2)
end

@testset "ispositive broadcasting" begin
    @test all(ispositive.(OneZero(), [1]))
    @test all(ispositive.(OneTwo(), [1]))
    @test all(ispositive.(OneMinusOne(), [1]))
    @test all(ispositive.(OneVsRest(1, [2,3]), [1]))
    @test all(ispositive.(RestVsOne([1,2], 3), [1]))
    @test all(ispositive.(RestVsOne([1,2], 3), [2]))
end

@testset "isnegative" begin
    @test isnegative(OneZero(), 0)
    @test isnegative(OneTwo(), 2)
    @test isnegative(OneMinusOne(), -1)
    @test isnegative(OneVsRest(1, [2,3]), 2)
    @test isnegative(OneVsRest(1, [2,3]), 3)
    @test isnegative(RestVsOne([1,2], 3), 3)
end

@testset "isnegative broadcasting" begin
    @test all(isnegative.(OneZero(), [0]))
    @test all(isnegative.(OneTwo(), [2]))
    @test all(isnegative.(OneMinusOne(), [-1]))
    @test all(isnegative.(OneVsRest(1, [2,3]), [2]))
    @test all(isnegative.(OneVsRest(1, [2,3]), [3]))
    @test all(isnegative.(RestVsOne([1,2], 3), [3]))
end

@testset "positive_label" begin
    @test positive_label(OneZero()) == 1
    @test positive_label(OneTwo()) == 1
    @test positive_label(OneMinusOne()) == 1
    @test positive_label(OneVsRest(1, [2,3])) == 1
    @test positive_label(RestVsOne([1,2], 3)) == 1
    @test positive_label(RestVsOne([2,1], 3)) == 2
end

@testset "negative_label" begin
    @test negative_label(OneZero()) == 0
    @test negative_label(OneTwo()) == 2
    @test negative_label(OneMinusOne()) == -1
    @test negative_label(OneVsRest(1, [2,3])) == 2
    @test negative_label(OneVsRest(1, [3,2])) == 3
    @test negative_label(RestVsOne([1,2], 3)) == 3
end


@testset "check_encoding" begin
    @test check_encoding(OneZero(), rand([0,1], 1000))
    @test !check_encoding(OneZero(), rand([0,1,2], 1000))

    @test check_encoding(OneTwo(), rand([1,2], 1000))
    @test !check_encoding(OneTwo(), rand([1,2,3], 1000))

    @test check_encoding(OneMinusOne(), rand([1,-1], 1000))
    @test !check_encoding(OneMinusOne(), rand([1,-1, 2], 1000))

    @test check_encoding(OneVsOne(3, 4), rand([3,4], 1000))
    @test !check_encoding(OneVsOne(3, 4), rand([1,2], 1000))
    @test check_encoding(OneVsOne(:three, :four), rand([:three, :four], 1000))
    @test !check_encoding(OneVsOne(:three, :four), rand([:one, :two], 1000))
    @test check_encoding(OneVsOne("three", "four"), rand(["three", "four"], 1000))
    @test !check_encoding(OneVsOne("three", "four"), rand(["one", "two"], 1000))

    @test check_encoding(OneVsRest(1, [2,3]), rand([1,2,3], 1000))
    @test !check_encoding(OneVsRest(1, [2,3]), rand([1,2,3,4], 1000))
    @test check_encoding(OneVsRest(:one, [:two, :three]), rand([:one, :two, :three], 1000))
    @test !check_encoding(OneVsRest(:one, [:two, :three]), rand([:four, :five], 1000))
    @test check_encoding(OneVsRest("one", ["two", "three"]), rand(["one", "two", "three"], 1000))
    @test !check_encoding(OneVsRest("one", ["two", "three"]), rand(["four", "five"], 1000))

    @test check_encoding(RestVsOne([1,2], 3), rand([1,2,3], 1000))
    @test !check_encoding(RestVsOne([1,2], 3), rand([1,2,3,4], 1000))
    @test check_encoding(RestVsOne([:one, :two], :three), rand([:one, :two, :three], 1000))
    @test !check_encoding(RestVsOne([:one, :two], :three), rand([:four, :five], 1000))
    @test check_encoding(RestVsOne(["one", "two"], "three"), rand(["one", "two", "three"], 1000))
    @test !check_encoding(RestVsOne(["one", "two"], "three"), rand(["four", "five"], 1000))
end


@testset "recode" begin
    @test recode.(OneZero(), OneMinusOne(), [0,1,0,0,1]) == [-1,1,-1,-1,1]
    @test recode.(OneZero(), OneTwo(), [0,1,0,0,1]) == [2,1,2,2,1]
    @test recode.(OneZero(), OneVsRest(2, [3,4,5]), [0,1,0,0,1]) == [3,2,3,3,2]
    @test recode.(OneZero(), OneVsRest(:two, [:three,:four,:five]), [0,1,0,0,1]) == [:three,:two,:three,:three,:two]
    @test recode.(OneZero(), OneVsRest("two", ["three","four","five"]), [0,1,0,0,1]) == ["three","two","three","three","two"]
    @test recode.(OneZero(), RestVsOne([2,3,4], 5), [0,1,0,0,1]) == [5,2,5,5,2]
    @test recode.(OneZero(), RestVsOne([:two, :three,:four], :five), [0,1,0,0,1]) == [:five,:two,:five,:five,:two]
    @test recode.(OneZero(), RestVsOne(["two","three","four"], "five"), [0,1,0,0,1]) == ["five","two","five","five","two"]
    
    @test recode.(OneMinusOne(), OneZero(), [-1,1,-1,-1,1]) == [0,1,0,0,1]
    @test recode.(OneMinusOne(), OneTwo(), [-1,1,-1,-1,1]) == [2,1,2,2,1]
    @test recode.(OneMinusOne(), OneVsRest(2, [3,4,5]), [-1,1,-1,-1,1]) == [3,2,3,3,2]
    @test recode.(OneMinusOne(), OneVsRest(:two, [:three,:four,:five]), [-1,1,-1,-1,1]) == [:three,:two,:three,:three,:two]
    @test recode.(OneMinusOne(), OneVsRest("two", ["three","four","five"]), [-1,1,-1,-1,1]) == ["three","two","three","three","two"]
    @test recode.(OneMinusOne(), RestVsOne([2,3,4], 5), [-1,1,-1,-1,1]) == [5,2,5,5,2]
    @test recode.(OneMinusOne(), RestVsOne([:two, :three,:four], :five), [-1,1,-1,-1,1]) == [:five,:two,:five,:five,:two]
    @test recode.(OneMinusOne(), RestVsOne(["two","three","four"], "five"), [-1,1,-1,-1,1]) == ["five","two","five","five","two"]

    @test recode.(OneTwo(), OneZero(), [2,1,2,2,1]) == [0,1,0,0,1]
    @test recode.(OneTwo(), OneMinusOne(), [2,1,2,2,1]) == [-1,1,-1,-1,1]
    @test recode.(OneTwo(), OneVsRest(2, [3,4,5]), [2,1,2,2,1]) == [3,2,3,3,2]
    @test recode.(OneTwo(), OneVsRest(:two, [:three,:four,:five]), [2,1,2,2,1]) == [:three,:two,:three,:three,:two]
    @test recode.(OneTwo(), OneVsRest("two", ["three","four","five"]), [2,1,2,2,1]) == ["three","two","three","three","two"]
    @test recode.(OneTwo(), RestVsOne([2,3,4], 5), [2,1,2,2,1]) == [5,2,5,5,2]
    @test recode.(OneTwo(), RestVsOne([:two, :three,:four], :five), [2,1,2,2,1]) == [:five,:two,:five,:five,:two]
    @test recode.(OneTwo(), RestVsOne(["two","three","four"], "five"), [2,1,2,2,1]) == ["five","two","five","five","two"]

    @test recode.(OneVsOne(3, 4), OneZero(), [4,3,4,4,3]) == [0,1,0,0,1]
    @test recode.(OneVsOne(3, 4), OneMinusOne(), [4,3,4,4,3]) == [-1,1,-1,-1,1]
    @test recode.(OneVsOne(3, 4), OneTwo(), [4,3,4,4,3]) == [2,1,2,2,1]
    @test recode.(OneVsOne(3, 4), OneVsRest(1, [2, 3]), [4,3,4,4,3]) == [2,1,2,2,1]
    @test recode.(OneVsOne(3, 4), RestVsOne([2,3,4], 5), [4,3,4,4,3]) == [5,2,5,5,2]

    @test recode.(OneVsOne(:three, :four), OneZero(), [:four,:three,:four,:four,:three]) == [0,1,0,0,1]
    @test recode.(OneVsOne(:three, :four), OneMinusOne(), [:four,:three,:four,:four,:three]) == [-1,1,-1,-1,1]
    @test recode.(OneVsOne(:three, :four), OneTwo(), [:four,:three,:four,:four,:three]) == [2,1,2,2,1]
    @test recode.(OneVsOne(:three, :four), OneVsRest(1, [2, 3]), [:four,:three,:four,:four,:three]) == [2,1,2,2,1]
    @test recode.(OneVsOne(:three, :four), RestVsOne([2,3,4], 5), [:four,:three,:four,:four,:three]) == [5,2,5,5,2]

    @test recode.(OneVsOne("three", "four"), OneZero(), ["four","three","four","four","three"]) == [0,1,0,0,1]
    @test recode.(OneVsOne("three", "four"), OneMinusOne(), ["four","three","four","four","three"]) == [-1,1,-1,-1,1]
    @test recode.(OneVsOne("three", "four"), OneTwo(), ["four","three","four","four","three"]) == [2,1,2,2,1]
    @test recode.(OneVsOne("three", "four"), OneVsRest(1, [2, 3]), ["four","three","four","four","three"]) == [2,1,2,2,1]
    @test recode.(OneVsOne("three", "four"), RestVsOne([2,3,4], 5), ["four","three","four","four","three"]) == [5,2,5,5,2]

    @test recode.(OneVsRest(2, [3,4,5]), OneZero(), [3,2,4,5,2]) == [0,1,0,0,1]
    @test recode.(OneVsRest(2, [3,4,5]), OneMinusOne(), [3,2,4,5,2]) == [-1,1,-1,-1,1]
    @test recode.(OneVsRest(2, [3,4,5]), OneTwo(), [3,2,4,5,2]) == [2,1,2,2,1]
    @test recode.(OneVsRest(2, [3,4,5]), OneVsRest(1, [2, 3]), [3,2,4,5,2]) == [2,1,2,2,1]
    @test recode.(OneVsRest(2, [3,4,5]), RestVsOne([2,3,4], 5), [3,2,4,5,2]) == [5,2,5,5,2]

    @test recode.(OneVsRest(:two, [:three,:four,:five]), OneZero(), [:three,:two,:four,:five,:two]) == [0,1,0,0,1]
    @test recode.(OneVsRest(:two, [:three,:four,:five]), OneMinusOne(), [:three,:two,:four,:five,:two]) == [-1,1,-1,-1,1]
    @test recode.(OneVsRest(:two, [:three,:four,:five]), OneTwo(), [:three,:two,:four,:five,:two]) == [2,1,2,2,1]
    @test recode.(OneVsRest(:two, [:three,:four,:five]), OneVsRest(1, [2, 3]), [:three,:two,:four,:five,:two]) == [2,1,2,2,1]
    @test recode.(OneVsRest(:two, [:three,:four,:five]), RestVsOne([2,3,4], 5), [:three,:two,:four,:five,:two]) == [5,2,5,5,2]

    @test recode.(OneVsRest("two", ["three","four","five"]), OneZero(), ["three","two","four","five","two"]) == [0,1,0,0,1]
    @test recode.(OneVsRest("two", ["three","four","five"]), OneMinusOne(), ["three","two","four","five","two"]) == [-1,1,-1,-1,1]
    @test recode.(OneVsRest("two", ["three","four","five"]), OneTwo(), ["three","two","four","five","two"]) == [2,1,2,2,1]
    @test recode.(OneVsRest("two", ["three","four","five"]), OneVsRest(1, [2, 3]), ["three","two","four","five","two"]) == [2,1,2,2,1]
    @test recode.(OneVsRest("two", ["three","four","five"]), RestVsOne([2,3,4], 5), ["three","two","four","five","two"]) == [5,2,5,5,2]

    @test recode.(RestVsOne([2,3,4], 5), OneZero(), [5,2,5,5,3]) == [0,1,0,0,1]
    @test recode.(RestVsOne([2,3,4], 5), OneMinusOne(), [5,2,5,5,3]) == [-1,1,-1,-1,1]
    @test recode.(RestVsOne([2,3,4], 5), OneTwo(), [5,2,5,5,3]) == [2,1,2,2,1]
    @test recode.(RestVsOne([2,3,4], 5), OneVsRest(1, [2, 3]), [5,2,5,5,3]) == [2,1,2,2,1]
    @test recode.(RestVsOne([2,3,4], 5), RestVsOne([1,2,3], 4), [5,2,5,5,3]) == [4,1,4,4,1]

    @test recode.(RestVsOne([:two, :three,:four], :five), OneZero(), [:five,:two,:five,:five,:three]) == [0,1,0,0,1]
    @test recode.(RestVsOne([:two, :three,:four], :five), OneMinusOne(), [:five,:two,:five,:five,:three]) == [-1,1,-1,-1,1]
    @test recode.(RestVsOne([:two, :three,:four], :five), OneTwo(), [:five,:two,:five,:five,:three]) == [2,1,2,2,1]
    @test recode.(RestVsOne([:two, :three,:four], :five), OneVsRest(1, [2, 3]), [:five,:two,:five,:five,:three]) == [2,1,2,2,1]
    @test recode.(RestVsOne([:two, :three,:four], :five), RestVsOne([1,2,3], 4), [:five,:two,:five,:five,:three]) == [4,1,4,4,1]

    @test recode.(RestVsOne(["two","three","four"], "five"), OneZero(), ["five","two","five","five","three"]) == [0,1,0,0,1]
    @test recode.(RestVsOne(["two","three","four"], "five"), OneMinusOne(), ["five","two","five","five","three"]) == [-1,1,-1,-1,1]
    @test recode.(RestVsOne(["two","three","four"], "five"), OneTwo(), ["five","two","five","five","three"]) == [2,1,2,2,1]
    @test recode.(RestVsOne(["two","three","four"], "five"), OneVsRest(1, [2, 3]), ["five","two","five","five","three"]) == [2,1,2,2,1]
    @test recode.(RestVsOne(["two","three","four"], "five"), RestVsOne([1,2,3], 4), ["five","two","five","five","three"]) == [4,1,4,4,1]
end