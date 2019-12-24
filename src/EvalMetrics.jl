module EvalMetrics

import Base: show, precision
import Statistics: quantile
import StatsBase: RealVector, IntegerVector

export 
    # confusion matrix
    Counts,
    counts,

    # performance metrics
    true_positive,
    true_negative,
    false_positive,
    false_negative,
    true_positive_rate,
    sensitivity,
    recall,
    hit_rate,
    true_negative_rate,
    specificity,
    selectivity,
    false_positive_rate,
    fall_out,
    false_negative_rate,
    miss_rate,
    positive_predictive_value,
    precision,
    negative_predictive_value,
    false_discovery_rate,
    false_omission_rate,
    threat_score,
    critical_success_index,
    accuracy,
    balanced_accuracy,
    f1_score,
    fβ_score,
    matthews_correlation_coefficient,
    mcc,
    quantile,
    positive_likelihood_ratio,
    negative_likelihood_ratio,
    diagnostic_odds_ratio,

    # threshold functions
    thresholds,
    threshold_at_tpr,
    threshold_at_tnr,
    threshold_at_fpr,
    threshold_at_fnr,
    threshold_at_k,

    # curves
    curve,
    auc,
    roc_curve,
    pr_curve,
    pquant_curve

include("confusion_matrix.jl")
include("metrics.jl")
include("thresholds.jl")
include("curves.jl")

end