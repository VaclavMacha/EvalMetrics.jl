function test_thresholds(; atol::Real = 1e-2)
    n      = 10000
    scores = rand(n)
    target = rand(0:1, n)
    rate   = 0.4
    k      = 10

    types = [Bool, identity, string, Symbol]

    @testset "threshold_at_* with labeltype = $type" for type in types
        test_thresholds(type.(target), scores, rate; classes = (type(0), type(1)))
        test_thresholds(type.(target), scores, rate; classes = (type(0), [type(1)]))
        test_thresholds(type.(target), scores, rate; classes = ([type(0)], type(1)))
        test_thresholds(type.(target), scores, rate; classes = ([type(0)], [type(1)]))
    end

    @testset "threshold_at_k" begin
        scores_sorted = sort(scores)
        @test threshold_at_k(scores, k) == scores_sorted[end - k + 1]
        @test threshold_at_k(scores, k; rev = false) == scores_sorted[k]
    end
end


function test_thresholds(target, scores, rate; classes::Tuple = (0,1))
    @testset "threshold_at_tpr" begin
        t1      = threshold_at_tpr(target, scores, rate; classes = classes)
        tpr     = true_positive_rate(target, scores, t1; classes = classes)
        tpr_eps = true_positive_rate(target, scores, t1 + eps(); classes = classes)
        @test tpr >= rate > tpr_eps
    end

    @testset "threshold_at_tnr" begin
        t2      = threshold_at_tnr(target, scores, rate; classes = classes)
        tnr     = true_negative_rate(target, scores, t2; classes = classes)
        tnr_eps = true_negative_rate(target, scores, t2 - eps(); classes = classes)
        @test tnr >= rate > tnr_eps
    end

    @testset "threshold_at_fpr" begin
        t3      = threshold_at_fpr(target, scores, rate; classes = classes)
        fpr     = false_positive_rate(target, scores, t3; classes = classes)
        fpr_eps = false_positive_rate(target, scores, t3 - eps(); classes = classes)
        @test fpr <= rate < fpr_eps
    end

    @testset "threshold_at_fnr" begin
        t4      = threshold_at_fnr(target, scores, rate; classes = classes)
        fnr     = false_negative_rate(target, scores, t4; classes = classes)
        fnr_eps = false_negative_rate(target, scores, t4 + eps(); classes = classes)
        @test fnr <= rate < fnr_eps
    end
end