function test_thresholds(; atol::Real = 1e-2)
    n      = 10000
    scores = rand(n)
    target = rand(0:1, n)
    rate   = 0.4
    k      = 10

    @testset "threshold_at_tpr" begin
        t1 = threshold_at_tpr(target, scores, rate)
        @test true_positive_rate(target, scores, t1)         >= rate
        @test true_positive_rate(target, scores, t1 + eps()) <  rate
    end

    @testset "threshold_at_tnr" begin
        t2 = threshold_at_tnr(target, scores, rate)
        @test true_negative_rate(target, scores, t2)         >= rate
        @test true_negative_rate(target, scores, t2 - eps()) <  rate
    end

    @testset "threshold_at_fpr" begin
        t3 = threshold_at_fpr(target, scores, rate)
        @test false_positive_rate(target, scores, t3)         <= rate
        @test false_positive_rate(target, scores, t3 - eps()) >  rate
    end

    @testset "threshold_at_fnr" begin
        t4 = threshold_at_fnr(target, scores, rate)
        @test false_negative_rate(target, scores, t4)         <= rate
        @test false_negative_rate(target, scores, t4 + eps()) >  rate
    end

    @testset "threshold_at_k" begin
        scores_sorted = sort(scores)
        @test threshold_at_k(scores, k) == scores_sorted[end - k + 1]
        @test threshold_at_k(scores, k; rev = false) == scores_sorted[k]
    end
end