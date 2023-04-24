module EvalMetrics

using LinearAlgebra
using PrettyTables
using Statistics

import Base: show, precision

include("confusion_matrix.jl")
include("metrics.jl")

export AbstractConfusionMatrix, BinaryConfusionMatrix, ConfusionMatrix, confusion

export negatives,
    positives,
    true_positives,
    false_positives,
    true_negatives,
    false_negatives,
    true_positive_rate, sensitivity, recall, hit_rate,
    true_negative_rate, specificity, selectivity,
    false_positive_rate, fall_out, type_I_error,
    false_negative_rate, miss_rate, type_II_error,
    precision, positive_predictive_value,
    negative_predictive_value,
    false_discovery_rate,
    false_omission_rate,
    threat_score, critical_success_index,
    fβ_score,
    matthews_correlation_coefficient, mcc,
    quant,
    topquant,
    positive_likelihood_ratio,
    negative_likelihood_ratio,
    diagnostic_odds_ratio,
    prevalence,
    accuracy,
    balanced_accuracy,
    error_rate,
    balanced_error_rate

end
