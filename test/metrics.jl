function test_metric(metric, enc, cm, targets, predicts, scores, thres, val_true; kwargs...)
    @testset "$(metric)" begin
        @test metric(cm; kwargs...) == val_true
        @test metric([cm]; kwargs...) == [val_true]
        @test metric(targets, predicts; kwargs...) == val_true
        @test metric(enc, targets, predicts; kwargs...) == val_true
        @test metric(targets, scores, thres; kwargs...) == val_true
        @test metric(enc, targets, scores, thres; kwargs...) == val_true
        @test metric(targets, scores, [thres]; kwargs...) == [val_true]
        @test metric(enc, targets, scores, [thres]; kwargs...) == [val_true]
    end
end 


n = 1000
targets = rand(0:1, n)
scores = rand(n)
thres = 0.7
predicts = scores .>= thres

βs = [1,2,3,4]

p  = sum(targets .== 1)
n  = sum(targets .== 0)
tp = sum(targets .* predicts)
tn = sum((1 .- targets) .* (1 .- predicts))
fp = sum((1 .- targets) .* predicts)
fn = sum(targets .* (1 .- predicts))

cm = ConfusionMatrix(p,n,tp,tn,fp,fn)

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


@testset "Metrics for $enc encoding" for enc in encs
    global targets, predicts

    targets = recode.(current_encoding(), enc, targets)
    predicts = recode.(current_encoding(), enc, predicts)
    set_encoding(enc)

    # true positie
    test_metric(true_positive, enc, cm, targets, predicts, scores, thres, tp)

    # true negtives
    test_metric(true_negative, enc, cm, targets, predicts, scores, thres, tn)

    # false positive
    test_metric(false_positive, enc, cm, targets, predicts, scores, thres, fp)

    # false negative
    test_metric(false_negative, enc, cm, targets, predicts, scores, thres, fn)

    # true positie rate
    val_true = tp/p
    test_metric(true_positive_rate, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(sensitivity, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(recall, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(hit_rate, enc, cm, targets, predicts, scores, thres, val_true)

    # true negtives
    val_true = tn/n
    test_metric(true_negative_rate, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(specificity, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(selectivity, enc, cm, targets, predicts, scores, thres, val_true)

    # false positive
    val_true = fp/n
    test_metric(false_positive_rate, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(fall_out, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(type_I_error, enc, cm, targets, predicts, scores, thres, val_true)

    # false negative
    val_true = fn/p
    test_metric(false_negative_rate, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(miss_rate, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(type_II_error, enc, cm, targets, predicts, scores, thres, val_true)

    # precision
    val_true = tp/(tp + fp)
    if isnan(val_true)
        val_true = one(val_true)
    end
    test_metric(precision, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(positive_predictive_value, enc, cm, targets, predicts, scores, thres, val_true)

    # negative predictive value
    val_true = tn/(tn + fn)
    test_metric(negative_predictive_value, enc, cm, targets, predicts, scores, thres, val_true)

    # false discovery rate
    val_true = fp/(fp + tp)
    test_metric(false_discovery_rate, enc, cm, targets, predicts, scores, thres, val_true)

    # false omission rate
     val_true = fn/(fn + tn)
    test_metric(false_omission_rate, enc, cm, targets, predicts, scores, thres, val_true)

    # threat score
    val_true = tp/(tp + fn + fp)
    test_metric(threat_score, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(critical_success_index, enc, cm, targets, predicts, scores, thres, val_true)

    # accuracy
    val_true = (tp + tn)/(p + n)
    test_metric(accuracy, enc, cm, targets, predicts, scores, thres, val_true)

    # balanced accuracy
    val_true = (tp/p + tn/n)/2
    test_metric(balanced_accuracy, enc, cm, targets, predicts, scores, thres, val_true)

    # f1 score
    prec = tp/(tp + fp)
    if isnan(prec)
        prec = one(prec)
    end
    rec = tp/p
    val_true = 2*prec*rec/(prec + rec)
    test_metric(f1_score, enc, cm, targets, predicts, scores, thres, val_true)

    # fβ score
    @testset "fβ score: β = $(β)" for β in βs
        val_true = (1 + β^2)*prec*rec/(β^2*prec + rec)
        test_metric(fβ_score, enc, cm, targets, predicts, scores, thres, val_true; β = β)
    end

    # Matthews correlation coefficient
    val_true = (tp*tn - fp*fn)/sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
    test_metric(matthews_correlation_coefficient, enc, cm, targets, predicts, scores, thres, val_true)
    test_metric(mcc, enc, cm, targets, predicts, scores, thres, val_true)

    # quant
    val_true = (fn + tn)/(p + n)
    test_metric(quant, enc, cm, targets, predicts, scores, thres, val_true)

    # topquant
    val_true = 1 - (fn + tn)/(p + n)
    test_metric(topquant, enc, cm, targets, predicts, scores, thres, val_true)

    # positive likelihood ratio
    val_true = tp/p/(fp/n)
    test_metric(positive_likelihood_ratio, enc, cm, targets, predicts, scores, thres, val_true)

    # negative likelihood ratio
    val_true = fn/p/(tn/n)
    test_metric(negative_likelihood_ratio, enc, cm, targets, predicts, scores, thres, val_true)

    # diagnostic odds ratio
    val_true = (tp/p)*(tn/n)/((fp/n)*(fn/p))
    test_metric(diagnostic_odds_ratio, enc, cm, targets, predicts, scores, thres, val_true)

    # prevalence
    val_true = p/(p + n)
    test_metric(prevalence, enc, cm, targets, predicts, scores, thres, val_true)
end