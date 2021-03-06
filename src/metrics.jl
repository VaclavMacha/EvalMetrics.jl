abstract type AbstractMetric end

apply(::Type{M}, args...; kwargs...) where {M <: AbstractMetric} =
    apply(M, ConfusionMatrix(args...); kwargs...)

apply(::Type{M}, x::AbstractArray{<:ConfusionMatrix}; kwargs...) where {M <: AbstractMetric} =
    apply.(M, x; kwargs...)


macro metric(name)
    name_lw = Symbol(lowercase(string(name)))

    quote
        abstract type $(esc(name)) <: AbstractMetric end

        Base.@__doc__  function $(esc(name_lw))(args...; kwargs...)
            apply($(esc(name)), args...; kwargs...)
        end
    end
end


"""
    $(SIGNATURES)

Returns # true positive samples.
"""
@metric True_positive
apply(::Type{True_positive}, x::ConfusionMatrix) = x.tp


"""
    $(SIGNATURES)

Returns # true negative samples.
"""
@metric True_negative
apply(::Type{True_negative}, x::ConfusionMatrix) = x.tn


"""
    $(SIGNATURES)

Returns # false positive samples.
"""
@metric False_positive
apply(::Type{False_positive}, x::ConfusionMatrix) = x.fp


"""
    $(SIGNATURES)

Returns # false negative samples.
"""
@metric False_negative
apply(::Type{False_negative}, x::ConfusionMatrix) = x.fn


"""
    $(SIGNATURES)

Returns true positive rate `tp/p`.
Aliases: `sensitivity`,  `recall`, `hit_rate`.
"""
@metric True_positive_rate
apply(::Type{True_positive_rate}, x::ConfusionMatrix) = x.tp/x.p

const sensitivity = true_positive_rate
const recall      = true_positive_rate
const hit_rate    = true_positive_rate


"""
    $(SIGNATURES)

Returns true negative rate `tn/n`.
Aliases: `specificity`,  `selectivity`.
"""
@metric True_negative_rate
apply(::Type{True_negative_rate}, x::ConfusionMatrix) = x.tn/x.n

const specificity = true_negative_rate
const selectivity = true_negative_rate


"""
    $(SIGNATURES)

Returns false positive rate `fp/n`.
Aliases: `fall_out`, `type_I_error`.
"""
@metric False_positive_rate
apply(::Type{False_positive_rate}, x::ConfusionMatrix) = x.fp/x.n

const fall_out     = false_positive_rate
const type_I_error = false_positive_rate


"""
    $(SIGNATURES)

Returns false negative rate `fn/p`.
Aliases: `miss_rate`, `type_II_error`.
"""
@metric False_negative_rate
apply(::Type{False_negative_rate}, x::ConfusionMatrix) = x.fn/x.p

const miss_rate     = false_negative_rate
const type_II_error = false_negative_rate


"""
    $(SIGNATURES)

Returns precision `tp/(tp + fp)`.
Aliases: `positive_predictive_value`.
"""
@metric Precision
apply(::Type{Precision}, x::ConfusionMatrix) = (val = x.tp/(x.tp + x.fp); isnan(val) ? one(val) : val)

const positive_predictive_value = precision


"""
    $(SIGNATURES)

Returns negative predictive value `tn/(tn + fn)`.
"""
@metric Negative_predictive_value
apply(::Type{Negative_predictive_value}, x::ConfusionMatrix) = x.tn/(x.tn + x.fn)


"""
    $(SIGNATURES)

Returns false discovery rate `fp/(fp + tp)`.
"""
@metric False_discovery_rate
apply(::Type{False_discovery_rate}, x::ConfusionMatrix) = x.fp/(x.fp + x.tp)


"""
    $(SIGNATURES)

Returns false omission rate `fn/(fn + tn)`.
"""
@metric False_omission_rate
apply(::Type{False_omission_rate}, x::ConfusionMatrix) = x.fn/(x.fn + x.tn)


"""
    $(SIGNATURES)

Returns threat score `tp/(tp + fn + fp)`.
Aliases: `critical_success_index`.
"""
@metric Threat_score
apply(::Type{Threat_score}, x::ConfusionMatrix) = x.tp/(x.tp + x.fn + x.fp)

const critical_success_index = threat_score


"""
    $(SIGNATURES)

Returns accuracy `(tp + tn)/(p + n).
"""
@metric Accuracy
apply(::Type{Accuracy}, x::ConfusionMatrix) = (x.tp + x.tn)/(x.p + x.n)


"""
    $(SIGNATURES)

Returns balanced accuracy `(tpr + tnr)/2`.
"""
@metric Balanced_accuracy
apply(::Type{Balanced_accuracy}, x::ConfusionMatrix) =
    (true_positive_rate(x) + true_negative_rate(x))/2


"""
    $(SIGNATURES)

Returns error rate `1 - accuracy`.
"""
@metric Error_rate
apply(::Type{Error_rate}, x::ConfusionMatrix) = 1 - accuracy(x)


"""
    $(SIGNATURES)

Returns balanced error rate `1 - balanced_accuracy`.
"""
@metric Balanced_error_rate
apply(::Type{Balanced_error_rate}, x::ConfusionMatrix) = 1 - balanced_accuracy(x)


"""
    $(SIGNATURES)

Returns f1 score `2*precision*recall/(precision + recall)`.
"""
@metric F1_score
apply(::Type{F1_score}, x::ConfusionMatrix) =
    2*precision(x)*recall(x)/(precision(x) + recall(x))


"""
    $(SIGNATURES)

Returns fβ score `(1 + β^2)*precision*recall/(β^2*precision + recall)`.
"""
@metric Fβ_score
apply(::Type{Fβ_score}, x::ConfusionMatrix; β::Real = 1) =
    (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))


"""
    $(SIGNATURES)

Returns Matthews correlation coefficient `(tp*tn - fp*fn)/sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))`.
Aliases: ` mcc`.
"""
@metric Matthews_correlation_coefficient
apply(::Type{Matthews_correlation_coefficient}, x::ConfusionMatrix) =
    (x.tp*x.tn - x.fp*x.fn)/sqrt((x.tp + x.fp)*(x.tp + x.fn)*(x.tn + x.fp)*(x.tn + x.fn))

const mcc = matthews_correlation_coefficient


"""
    $(SIGNATURES)

Returns quant `(fn + tn)/(p + n)`.
"""
@metric Quant
apply(::Type{Quant}, x::ConfusionMatrix) = (x.fn + x.tn)/(x.p + x.n)


"""
    $(SIGNATURES)

Returns topquant `1 - quant`.
"""
@metric Topquant
apply(::Type{Topquant}, x::ConfusionMatrix) = 1 - quant(x)


"""
    $(SIGNATURES)

Returns positive likelihood ratio `tpr/fpr`.
"""
@metric Positive_likelihood_ratio
apply(::Type{Positive_likelihood_ratio}, x::ConfusionMatrix) =
    true_positive_rate(x)/false_positive_rate(x)


"""
    $(SIGNATURES)

Returns negative likelihood ratio `fnr/tnr`.
"""
@metric Negative_likelihood_ratio
apply(::Type{Negative_likelihood_ratio}, x::ConfusionMatrix) =
    false_negative_rate(x)/true_negative_rate(x)


"""
    $(SIGNATURES)

Returns diagnostic odds ratio `tpr*tnr/(fpr*fnr)`.
"""
@metric Diagnostic_odds_ratio
apply(::Type{Diagnostic_odds_ratio}, x::ConfusionMatrix) =
    true_positive_rate(x)*true_negative_rate(x)/(false_positive_rate(x)*false_negative_rate(x))


"""
    $(SIGNATURES)

Returns prevalence `p/(p + n)`.
"""
@metric Prevalence
apply(::Type{Prevalence}, x::ConfusionMatrix) = x.p/(x.p + x.n)