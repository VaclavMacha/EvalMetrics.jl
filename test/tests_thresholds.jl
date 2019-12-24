function test_thresholds(; atol::Real = 1e-2)
    n      = 10000
    scores = sort(rand(n))
    target = rand(0:1, n)
    rate   = 0.3*rand() + 0.3
    k      = rand(1:n)

    @test true_positive_rate(target, scores, threshold_at_tpr(target, scores, rate))  ≈ rate atol = atol
    @test true_negative_rate(target, scores, threshold_at_tnr(target, scores, rate))  ≈ rate atol = atol
    @test false_positive_rate(target, scores, threshold_at_fpr(target, scores, rate)) ≈ rate atol = atol
    @test false_negative_rate(target, scores, threshold_at_fnr(target, scores, rate)) ≈ rate atol = atol

    @test true_positive_rate(target, scores, threshold_at_tpr(target, scores, rate))  >= rate
    @test true_negative_rate(target, scores, threshold_at_tnr(target, scores, rate))  >= rate
    @test false_positive_rate(target, scores, threshold_at_fpr(target, scores, rate)) <= rate
    @test false_negative_rate(target, scores, threshold_at_fnr(target, scores, rate)) <= rate

    @test threshold_at_k(scores, k) == scores[end - k + 1]
    @test threshold_at_k(scores, k; rev = false) == scores[k]
end