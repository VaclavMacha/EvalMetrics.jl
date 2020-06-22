module EvalMetrics


import Base: show, precision
import DocStringExtensions: SIGNATURES
import Statistics: quantile
import StatsBase: RealVector, IntegerVector
using RecipesBase
using Reexport

include("encodings/Encodings.jl")
@reexport using .Encodings

include("utilities.jl")
include("confusion_matrix.jl")

const CMVector{T<:Real} = AbstractVector{ConfusionMatrix{T}}

include("metrics.jl")
include("thresholds.jl")
include("curves.jl")

export 
    # confusion matrix
    ConfusionMatrix,

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
        type_I_error,
    false_negative_rate,
        miss_rate,
        type_II_error,
    precision,
        positive_predictive_value,
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
    quant,
    topquant,
    positive_likelihood_ratio,
    negative_likelihood_ratio,
    diagnostic_odds_ratio,
    prevalence,

    # threshold functions
    thresholds,
    threshold_at_tpr,
    threshold_at_tnr,
    threshold_at_fpr,
    threshold_at_fnr,
    threshold_at_k,

    #curves
    auc_trapezoidal, auc,
    PRCurve, prcurve, au_prcurve, prplot,
    ROCCurve, roccurve, au_roccurve, rocplot,

    # utilities
    binary_eval_report,
    mergesorted


binary_eval_report(target::AbstractVector, scores::RealVector, fpr = 0.05) =
    binary_eval_report(current_encoding(), target, scores, fpr)


function binary_eval_report(enc::TwoClassEncoding, target::AbstractVector, scores::RealVector, fpr = 0.05)
    t = threshold_at_fpr(enc, target, scores, fpr)
    c = ConfusionMatrix(enc, target, scores, t)
    
    return Dict(
        "accuracy@fpr$(fpr)" => accuracy(c),
        "au_prcurve" => au_prcurve(enc, target, scores),
        "au_roccurve" => au_roccurve(enc, target, scores),
        "precision@fpr$(fpr)" => precision(c),
        "prevalence" => prevalence(c),
        "recall@fpr$(fpr)" => recall(c),
        "samples" => length(target),
        "true negative rate@fpr$(fpr)" => true_negative_rate(c)
    )
end

end
