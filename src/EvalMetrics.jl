module EvalMetrics

const LabelType                 = Union{Bool, Real, String, Symbol}
const LabelVector{T<:LabelType} = AbstractArray{T,1}
const CountsVector{T<:Counts} = AbstractArray{T,1}
const CountsMatrix{T<:Counts} = AbstractArray{T, 2}
const CountsArray{T<:Counts}  = AbstractArray{T}

using  MacroTools
import MacroTools: combinedef
import Base: show, precision
import DocStringExtensions: SIGNATURES
import Statistics: quantile
import StatsBase: RealVector, IntegerVector

export 
    # confusion matrix
    Counts,
    counts,
    CountsVector,
    CountsMatrix,
    CountsArray,
    LabelType,
    LabelVector,
    RealVector,
    IntegerVector,

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
    auc,
    mergesorted

include("utilities.jl")
include("confusion_matrix.jl")
include("metrics.jl")
include("thresholds.jl")

end
