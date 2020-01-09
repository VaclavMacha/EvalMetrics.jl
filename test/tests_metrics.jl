function test_metrics()
    target  = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    predict = [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    scores  = [0.2, 0.8, 0.3, 0.9, 0.6, 0.7, 0.2, 0.3, 0.1, 1.1]
    thres   = 0.65
    correct = (p  = 5, n  = 5, tp = 3, tn = 4, fp = 1, fn = 2, β  = 1.1)

    types = [Bool, identity, string, Symbol]

    @testset "input = (target, predict) with labeltype = $type" for type in types
        test_metrics(correct, type.(target), type.(predict); classes = (type(0), type(1)))
        test_metrics(correct, type.(target), type.(predict); classes = (type(0), [type(1)]))
        test_metrics(correct, type.(target), type.(predict); classes = ([type(0)], type(1)))
        test_metrics(correct, type.(target), type.(predict); classes = ([type(0)], [type(1)]))
    end

    @testset "input = (target, scores, threshold) with labeltype = $type" for type in types
        test_metrics(correct, type.(target), scores, thres; classes = (type(0), type(1)))
        test_metrics(correct, type.(target), scores, thres; classes = (type(0), [type(1)]))
        test_metrics(correct, type.(target), scores, thres; classes = ([type(0)], type(1)))
        test_metrics(correct, type.(target), scores, thres; classes = ([type(0)], [type(1)]))
    end
end


function test_metrics(correct, input...; classes::Tuple = (0,1))
    p, n, tp, tn, fp, fn, β = correct 

    @test true_positive(input...; classes = classes)  == tp
    @test true_negative(input...; classes = classes)  == tn
    @test false_positive(input...; classes = classes) == fp
    @test false_negative(input...; classes = classes) == fn

    @testset "true positive rate and its aliases" begin
        @test true_positive_rate(input...; classes = classes)  == tp/p
        @test sensitivity(input...; classes = classes)         == tp/p
        @test recall(input...; classes = classes)              == tp/p
        @test hit_rate(input...; classes = classes)            == tp/p
    end

    @testset "true negative rate and its aliases" begin
        @test true_negative_rate(input...; classes = classes)  == tn/n
        @test specificity(input...; classes = classes)         == tn/n
        @test selectivity(input...; classes = classes)         == tn/n
    end

    @testset "false positive rate and its aliases" begin
        @test false_positive_rate(input...; classes = classes) == fp/n
        @test fall_out(input...; classes = classes)            == fp/n
        @test type_I_error(input...; classes = classes)        == fp/n
    end

    @testset "false negative rate and its aliases" begin
        @test false_negative_rate(input...; classes = classes) == fn/p
        @test miss_rate(input...; classes = classes)           == fn/p
        @test type_II_error(input...; classes = classes)       == fn/p
    end

    @testset "positive predictive value and its aliases" begin
        @test positive_predictive_value(input...; classes = classes) == tp/(tp + fp)
        @test precision(input...; classes = classes)                 == tp/(tp + fp)
    end

    @test negative_predictive_value(input...; classes = classes) == tn/(tn + fn)
    @test false_discovery_rate(input...; classes = classes)      == fp/(fp + tp)
    @test false_omission_rate(input...; classes = classes)       == fn/(fn + tn)

    @testset "threat score its aliases" begin
        @test threat_score(input...; classes = classes)           == tp/(tp + fn + fp)
        @test critical_success_index(input...; classes = classes) == tp/(tp + fn + fp)
    end

    @test accuracy(input...; classes = classes) == (tp + tn)/(p + n)
    @test balanced_accuracy(input...; classes = classes) == (tp/p + tn/n)/2

    @testset "f scores" begin
        prec = tp/(tp + fp)
        recl = tp/p
        @test f1_score(input...; classes = classes) ==
            2*(prec * recl)/(prec + recl)
        @test fβ_score(input...; β = β, classes = classes) ==
            (1 + β^2)*(prec * recl)/(β^2*prec + recl)
    end

    @testset "matthews correlation coefficient its aliases" begin
        @test matthews_correlation_coefficient(input...; classes = classes) ==
            (tp*tn + fp*fn)/sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
        @test mcc(input...; classes = classes) ==
            (tp*tn + fp*fn)/sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    end

    @test quant(input...; classes = classes) == (fn + tn)/(p + n)

    @test positive_likelihood_ratio(input...; classes = classes) == (tp/p)/(fp/n)
    @test negative_likelihood_ratio(input...; classes = classes) == (fn/p)/(tn/n)
    @test diagnostic_odds_ratio(input...; classes = classes)     == ((tp/p)/(fp/n))/((fn/p)/(tn/n))
end