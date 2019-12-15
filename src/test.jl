using Revise
using EvalMetrics

using BenchmarkTools
import EvalCurves


function test_evalcurves(y, s, y_pred, fpr, tpr, k, p)
    # p    = EvalCurves.precision_at_k(s, y, k)
    # t    = EvalCurves.tpr_at_fpr(fpr, tpr, p)
    acc  = EvalCurves.accuracy(y, y_pred)
    npv  = EvalCurves.negative_predictive_value(y, y_pred)
    fdr  = EvalCurves.false_discovery_rate(y, y_pred)
    fomr = EvalCurves.false_omission_rate(y, y_pred)
    mcc  = EvalCurves.matthews_correlation_coefficient(y, y_pred)

    return [acc, npv, fdr, fomr, mcc]
end

function test_evalmetrics(y, s, y_pred, fpr, tpr, k, p)
    rc = EvalMetrics.counts(y, y_pred)

    # p    = EvalMetrics.precision_at_k(y, s, k)
    # t    = EvalMetrics.tpr_at_fpr(fpr, tpr, p)
    acc  = EvalMetrics.accuracy(rc)
    npv  = EvalMetrics.negative_predictive_value(rc)
    fdr  = EvalMetrics.false_discovery_rate(rc)
    fomr = EvalMetrics.false_omission_rate(rc)
    mcc  = EvalMetrics.matthews_correlation_coefficient(rc)

    return [acc, npv, fdr, fomr, mcc]
end


m        = 100000;
y        = rand(0:1, m);
y_pred   = rand(0:1, m);
scores   = rand(m);
fpr, tpr = EvalCurves.roccurve(scores, y);

k = 5
p = 0.5

@time out1 = test_evalcurves(y, scores, y_pred, fpr, tpr, k, p);
@time out2 = test_evalmetrics(y, scores, y_pred, fpr, tpr, k, p);

[out1 out2]

@btime out1 = test_evalcurves(y, scores, y_pred, fpr, tpr, k, p);
@btime out2 = test_evalmetrics(y, scores, y_pred, fpr, tpr, k, p);



# m      = 10000;
# y      = rand(0:1, m);
# scores = rand(m);
# fpr = 0.5
# tnr = 0.45
# fnr = 0.55
# tpr = 0.40

# EvalCurves.false_positive_rate(y, Int.(scores .>= EvalCurves.threshold_at_fpr(scores, y, fpr)))

# (fpr, false_positive_rate(roc(y, scores, threshold_at_fpr(y, scores, fpr))))
# (tnr, true_negative_rate(roc(y, scores, threshold_at_tnr(y, scores, tnr))))
# (fnr, false_negative_rate(roc(y, scores, threshold_at_fnr(y, scores, fnr))))
# (tpr, true_positive_rate(roc(y, scores, threshold_at_tpr(y, scores, tpr))))

# true_positive_rate(roc(y, scores, threshold_at_fpr(y, scores, fpr)))

# fprv, tprv = EvalCurves.roccurve(scores, y);
# tpr_at_fpr(fprv, tprv, fpr)