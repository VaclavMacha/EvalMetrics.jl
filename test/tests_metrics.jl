function test_metrics()
    target  = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    predict = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    scores  = [0.2, 0.8, 0.3, 0.9, 0.6, 0.7, 0.2, 0.3, 0.1, 1.1]
    thres   = 0.65
    correct = (p  = 5, n  = 5, tp = 3, tn = 4, fp = 1, fn = 2, β  = 1.1)

    @testset "input = (target, predict)"  begin
        test_metrics(correct, target, predict)
    end

    @testset "input = (target, scores, threshold)" begin
        test_metrics(correct, target, scores, thres)
    end
end

function test_metrics(correct, input...)
    p, n, tp, tn, fp, fn, β = correct 

    @test true_positive(input...)  == tp
    @test true_negative(input...)  == tn
    @test false_positive(input...) == fp
    @test false_negative(input...) == fn

    @testset "true positive rate and its aliases" begin
        @test true_positive_rate(input...)  == tp/p
        @test sensitivity(input...)         == tp/p
        @test recall(input...)              == tp/p
        @test hit_rate(input...)            == tp/p
    end

    @testset "true negative rate and its aliases" begin
        @test true_negative_rate(input...)  == tn/n
        @test specificity(input...)         == tn/n
        @test selectivity(input...)         == tn/n
    end

    @testset "false positive rate and its aliases" begin
        @test false_positive_rate(input...) == fp/n
        @test fall_out(input...)            == fp/n
        @test type_I_error(input...)        == fp/n
    end

    @testset "false negative rate and its aliases" begin
        @test false_negative_rate(input...) == fn/p
        @test miss_rate(input...)           == fn/p
        @test type_II_error(input...)       == fn/p
    end

    @testset "positive predictive value and its aliases" begin
        @test positive_predictive_value(input...) == tp/(tp + fp)
        @test precision(input...)                 == tp/(tp + fp)
    end

    @test negative_predictive_value(input...) == tn/(tn + fn)
    @test false_discovery_rate(input...)      == fp/(fp + tp)
    @test false_omission_rate(input...)       == fn/(fn + tn)

    @testset "threat score its aliases" begin
        @test threat_score(input...)           == tp/(tp + fn + fp)
        @test critical_success_index(input...) == tp/(tp + fn + fp)
    end

    @test accuracy(input...) == (tp + tn)/(p + n)
    @test balanced_accuracy(input...) == (tp/p + tn/n)/2

    @testset "f scores" begin
        prec = tp/(tp + fp)
        recl = tp/p
        @test f1_score(input...) ==
            2*(prec * recl)/(prec + recl)
        @test fβ_score(input...; β = β) ==
            (1 + β^2)*(prec * recl)/(β^2*prec + recl)
    end

    @testset "matthews correlation coefficient its aliases" begin
        @test matthews_correlation_coefficient(input...) ==
            (tp*tn + fp*fn)/sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        @test mcc(input...) ==
            (tp*tn + fp*fn)/sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    end

    @test quant(input...) == (fn + tn)/(p + n)

    @test positive_likelihood_ratio(input...) == (tp/p)/(fp/n)
    @test negative_likelihood_ratio(input...) == (fn/p)/(tn/n)
    @test diagnostic_odds_ratio(input...)     == ((tp/p)/(fp/n))/((fn/p)/(tn/n))
end