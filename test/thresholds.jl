function test_true_rates(rate, val, val_eps)
    if iszero(rate) 
        val >= rate >= val_eps
    else
        val >= rate > val_eps
    end
end

function test_false_rates(rate, val, val_eps)
    if isone(rate) 
        val <= rate <= val_eps
    else
        val <= rate < val_eps
    end
end

n = 1000
targets = rand(0:1, n)
scores = rand(n)

# make non-unique scores
scores[rand(1:1000, 10)] .= 0
scores[rand(1:1000, 10)] .= 1
for k in 1:18
    scores[rand(1:1000, 10)] .= rand()
end

scores_sorted = sort(scores)
rates = collect(0:0.01:1)
rates = vcat(rates, [rates])
ks = 1:10  

set_encoding(OneZero())

encs = [
    OneZero(),
    OneMinusOne(),
    OneTwo(),
    OneVsOne(3,4),
    OneVsOne(:three,:four),
    OneVsOne("three","four"),
    OneVsRest(1, [2,3,4]),
    OneVsRest(:one, [:two, :three]),
    OneVsRest("one", ["two", "three"]),
    RestVsOne([1,2,3], 4),
    RestVsOne([:one, :two], :three),
    RestVsOne(["one", "two"], "three")
]


@testset "threshold_at_k: k = $k" for k in ks
    @test threshold_at_k(scores, k) == scores_sorted[end - k + 1]
    @test threshold_at_k(scores, k; rev = true) == scores_sorted[end - k + 1]
    @test threshold_at_k(scores, k; rev = false) == scores_sorted[k]
end



@testset "Thresholds for $enc encoding" for enc in encs
    global targets
    
    targets = recode.(current_encoding(), enc, targets)
    set_encoding(enc)

    @testset "rate = $(rate)" for rate in rates
        @testset "threshold_at_tpr" begin
            thres = threshold_at_tpr(targets, scores, rate)
            val = true_positive_rate(targets, scores, thres)
            val_eps = true_positive_rate(targets, scores, thres .+ eps.(thres))

            @test all(test_true_rates.(rate, val, val_eps))

            thres = threshold_at_tpr(enc, targets, scores, rate)
            tpr = true_positive_rate(enc, targets, scores, thres)
            tpr_eps = true_positive_rate(enc, targets, scores, thres .+ eps.(thres))

            @test all(test_true_rates.(rate, tpr, tpr_eps))
        end

        @testset "threshold_at_tnr" begin
            thres = threshold_at_tnr(targets, scores, rate)
            val = true_negative_rate(targets, scores, thres)
            val_eps = true_negative_rate(targets, scores, thres .- eps.(thres))

            @test all(test_true_rates.(rate, val, val_eps))

            thres = threshold_at_tnr(enc, targets, scores, rate)
            val = true_negative_rate(enc, targets, scores, thres)
            val_eps = true_negative_rate(enc, targets, scores, thres .- eps.(thres))

            @test all(test_true_rates.(rate, val, val_eps))
        end

        @testset "threshold_at_fpr" begin
            thres = threshold_at_fpr(targets, scores, rate)
            val = false_positive_rate(targets, scores, thres)
            val_eps = false_positive_rate(targets, scores, thres .- eps.(thres))

            @test all(test_false_rates.(rate, val, val_eps))

            thres = threshold_at_fpr(enc, targets, scores, rate)
            val = false_positive_rate(enc, targets, scores, thres)
            val_eps = false_positive_rate(enc, targets, scores, thres .- eps.(thres))

            @test all(test_false_rates.(rate, val, val_eps))
        end

        @testset "threshold_at_fnr" begin
            thres = threshold_at_fnr(targets, scores, rate)
            val = false_negative_rate(targets, scores, thres)
            val_eps = false_negative_rate(targets, scores, thres .+ eps.(thres))

            @test all(test_false_rates.(rate, val, val_eps))

            thres = threshold_at_fnr(enc, targets, scores, rate)
            val = false_negative_rate(enc, targets, scores, thres)
            val_eps = false_negative_rate(enc, targets, scores, thres .+ eps.(thres))

            @test all(test_false_rates.(rate, val, val_eps))
        end
    end
end