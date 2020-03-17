function test_thresholds(; atol::Real = 1e-2)
    n      = 100000
    scores = rand(n)
    target = rand(0:1, n)
    rates  = collect(0:0.01:1)
    k      = 10

    types = [Bool, identity, string, Symbol]

    @testset "threshold_at_* with labeltype = $type" for type in types
        test_thresholds(type.(target), scores, rates; classes = (type(0), type(1)))
        test_thresholds(type.(target), scores, rates; classes = (type(0), [type(1)]))
        test_thresholds(type.(target), scores, rates; classes = ([type(0)], type(1)))
        test_thresholds(type.(target), scores, rates; classes = ([type(0)], [type(1)]))
    end

    @testset "threshold_at_k" begin
        scores_sorted = sort(scores)
        @test threshold_at_k(scores, k) == scores_sorted[end - k + 1]
        @test threshold_at_k(scores, k; rev = false) == scores_sorted[k]
    end
end


function test_thresholds(target, scores, rates; classes::Tuple = (0,1))
    @testset "threshold_at_tpr" begin
        t1      = threshold_at_tpr(target, scores, rates; classes = classes)
        tpr     = true_positive_rate(target, scores, t1; classes = classes)
        tpr_eps = true_positive_rate(target, scores, t1 .+ eps(); classes = classes)
        @test all(tpr[2:end-1]  .>= rates[2:end-1]  .>  tpr_eps[2:end-1])
        @test all(tpr[[1, end]] .>= rates[[1, end]] .>= tpr_eps[[1, end]])
    end

    @testset "threshold_at_tnr" begin
        t2      = threshold_at_tnr(target, scores, rates; classes = classes)
        tnr     = true_negative_rate(target, scores, t2; classes = classes)
        tnr_eps = true_negative_rate(target, scores, t2 .- eps(); classes = classes)
        @test all(tnr[2:end-1]  .>= rates[2:end-1]  .>  tnr_eps[2:end-1])
        @test all(tnr[[1, end]] .>= rates[[1, end]] .>= tnr_eps[[1, end]])
    end

    @testset "threshold_at_fpr" begin
        t3      = threshold_at_fpr(target, scores, rates; classes = classes)
        fpr     = false_positive_rate(target, scores, t3; classes = classes)
        fpr_eps = false_positive_rate(target, scores, t3 .- eps(); classes = classes)
        @test all(fpr[2:end-1]  .<= rates[2:end-1]  .< fpr_eps[2:end-1])
        @test all(fpr[[1, end]] .<= rates[[1, end]] .<= fpr_eps[[1, end]])
    end

    @testset "threshold_at_fnr" begin
        t4      = threshold_at_fnr(target, scores, rates; classes = classes)
        fnr     = false_negative_rate(target, scores, t4; classes = classes)
        fnr_eps = false_negative_rate(target, scores, t4 .+ eps(); classes = classes)
        @test all(fnr[2:end-1]  .<= rates[2:end-1]  .< fnr_eps[2:end-1])
        @test all(fnr[[1, end]] .<= rates[[1, end]] .<= fnr_eps[[1, end]])
    end
end