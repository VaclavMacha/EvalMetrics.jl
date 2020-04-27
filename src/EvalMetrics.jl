module EvalMetrics

using  MacroTools
import MacroTools: combinedef
import Base: show, precision
import DocStringExtensions: SIGNATURES
import Statistics: quantile
import StatsBase: RealVector, IntegerVector

const LabelType = Union{Bool, Real, String, Symbol}
const LabelVector{T<:LabelType} = AbstractVector{T}

include("utilities.jl")
include("confusion_matrix.jl")

const CountVector{T<:Real} = AbstractVector{Counts{T}}

include("metrics.jl")
include("thresholds.jl")
include("curves.jl")

export 
    # confusion matrix
    Counts,
    counts,
    LabelType,
    LabelVector,

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

    # threshold functions
    thresholds,
    threshold_at_tpr,
    threshold_at_tnr,
    threshold_at_fpr,
    threshold_at_fnr,
    threshold_at_k,

    # utilities
    binary_eval_report,
    auc,
    auroc, auprc,
    mergesorted

function binary_eval_report(target::LabelVector, scores::RealVector, fpr=0.05; classes::Tuple=(0,1))
    t = threshold_at_fpr(target, scores, fpr; classes=classes)
    c = counts(target, scores, t; classes=classes)
    Dict(
         "accuracy@fpr$(fpr)" => accuracy(c),
         "auprc" => auprc(target, scores; classes=classes),
         "auroc" => auroc(target, scores; classes=classes),
         "precision@fpr$(fpr)" => precision(c),
         "prevalence" => (c.fn + c.tp)/length(target),
         "recall@fpr$(fpr)" => recall(c),
         "samples" => length(target),
         "true negative rate@fpr$(fpr)" => true_negative_rate(c)
        )
end

end
