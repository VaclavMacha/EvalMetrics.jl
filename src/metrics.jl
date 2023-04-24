abstract type AbstractMetric end

function apply(
    M::AbstractMetric,
    args...;
    metric_kwargs::NamedTuple=NamedTuple(),
    kwargs...
)
    return apply(M, confusion(args...; kwargs...); metric_kwargs...)
end

function apply(
    M::AbstractMetric,
    x::AbstractArray{<:AbstractConfusionMatrix};
    kwargs...
)

    return apply.(M, x; kwargs...)
end

function apply(M::AbstractMetric, C::AbstractConfusionMatrix; kwargs...)
    name_lw = lowercase(string(M.name.name))
    N = size(C, 2)
    type = N == 2 ? "binary" : "multi-class"
    error("$(name_lw) not defined for $(type) confusion matrix")
    return
end

# ------------------------------------------------------------------------------------------
# binary classification metrics
# ------------------------------------------------------------------------------------------
struct Negatives <: AbstractMetric end
apply(::Negatives, C::BinaryConfusionMatrix) = sum(C[1, :])
apply(::Negatives, C::ConfusionMatrix) = sum(C) .- vec(sum(C; dims=2))

"""
    negatives(C::BinaryConfusionMatrix)

Return the number of negative samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negatives(y, ŷ)
4
```
"""
negatives(args...) = apply(Negatives(), args...)

struct Positives <: AbstractMetric end
apply(::Positives, C::BinaryConfusionMatrix) = sum(C[2, :])
apply(::Positives, C::ConfusionMatrix) = vec(sum(C; dims=2))

"""
    positives(C::BinaryConfusionMatrix)

Return the number of positive samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> positives(y, ŷ)
6
```
"""
positives(args...) = apply(Positives(), args...)


struct True_Positives <: AbstractMetric end
apply(::True_Positives, C::BinaryConfusionMatrix) = C[2, 2]
apply(::True_Positives, C::ConfusionMatrix) = diag(C)

"""
    true_positives(C::BinaryConfusionMatrix)

Return the number of correctly classified positive samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_positives(y, ŷ)
3
```
"""
true_positives(args...) = apply(True_Positives(), args...)

struct False_Positives <: AbstractMetric end
apply(::False_Positives, C::BinaryConfusionMatrix) = C[1, 2]
apply(::False_Positives, C::ConfusionMatrix) = vec(sum(C; dims=1)) .- diag(C)

"""
    false_positives(C::BinaryConfusionMatrix)

Return the number of incorrectly classified negative samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_positives(y, ŷ)
2
```
"""
false_positives(args...) = apply(False_Positives(), args...)

struct True_Negatives <: AbstractMetric end
apply(::True_Negatives, C::AbstractConfusionMatrix) = negatives(C) .- false_positives(C)

"""
    true_negatives(C::BinaryConfusionMatrix)

Return the number of correctly classified negative samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_negatives(y, ŷ)
2
```
"""
true_negatives(args...) = apply(True_Negatives(), args...)

struct False_Negatives <: AbstractMetric end
apply(::False_Negatives, C::AbstractConfusionMatrix) = positives(C) .- true_positives(C)

"""
    false_negatives(C::BinaryConfusionMatrix)

Return the number of incorrectly classified positive samples.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negatives(y, ŷ)
3
```
"""
false_negatives(args...) = apply(False_Negatives(), args...)

struct TruePositiveRate <: AbstractMetric end
apply(::TruePositiveRate, C::AbstractConfusionMatrix) = true_positives(C) ./ positives(C)

@doc raw"""
    true_positive_rate(C::BinaryConfusionMatrix)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_positive\_rate} = \frac{tp}{p}
```

Can be also called via aliases `sensitivity`,  `recall`, `hit_rate`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negative_rate(y, ŷ)
0.5

julia> sensitivity(y, ŷ)
0.5

julia> recall(y, ŷ)
0.5

julia> hit_rate(y, ŷ)
0.5
```
"""
true_positive_rate(args...) = apply(TruePositiveRate(), args...)
const sensitivity = true_positive_rate
const recall = true_positive_rate
const hit_rate = true_positive_rate

struct TrueNegativeRate <: AbstractMetric end
apply(::TrueNegativeRate, C::AbstractConfusionMatrix) = true_negatives(C) ./ negatives(C)

@doc raw"""
    true_negative_rate(C::BinaryConfusionMatrix)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_negative\_rate} = \frac{tn}{n}
```

Can be also called via aliases `specificity`,  `selectivity`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_negative_rate(y, ŷ)
0.5

julia> specificity(y, ŷ)
0.5

julia> selectivity(y, ŷ)
0.5
```
"""
true_negative_rate(args...) = apply(TrueNegativeRate(), args...)
const specificity = true_negative_rate
const selectivity = true_negative_rate

struct FalsePositiveRate <: AbstractMetric end
apply(::FalsePositiveRate, C::AbstractConfusionMatrix) = false_positives(C) ./ negatives(C)

@doc raw"""
    false_positive_rate(C::BinaryConfusionMatrix)

Return the proportion of incorrectly classified negative samples, i.e

```math
\mathrm{false\_positive\_rate} = \frac{tn}{p}
```

Can be also called via aliases `fall_out`, `type_I_error`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_positive_rate(y, ŷ)
0.5

julia> fall_out(y, ŷ)
0.5

julia> type_I_error(y, ŷ)
0.5
```
"""
false_positive_rate(args...) = apply(FalsePositiveRate(), args...)
const fall_out = false_positive_rate
const type_I_error = false_positive_rate

struct FalseNegativeRate <: AbstractMetric end
apply(::FalseNegativeRate, C::AbstractConfusionMatrix) = false_negatives(C) ./ positives(C)

@doc raw"""
    false_negative_rate(C::BinaryConfusionMatrix)

Return the proportion of incorrectly classified positive samples, i.e

```math
\mathrm{false\_negative\_rate} = \frac{tp}{n}
```

Can be also called via aliases `miss_rate`, `type_II_error`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negative_rate(y, ŷ)
0.5

julia> miss_rate(y, ŷ)
0.5

julia> type_II_error(y, ŷ)
0.5
```
"""
false_negative_rate(args...) = apply(FalseNegativeRate(), args...)
const miss_rate = false_negative_rate
const type_II_error = false_negative_rate

struct Precision <: AbstractMetric end
function apply(::Precision, C::BinaryConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = tp / (tp + fp)
    return isnan(val) ? one(val) : val
end

function apply(::Precision, C::ConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = tp ./ (tp .+ fp)
    val[isnan.(val)] .= 1
    return val
end


@doc raw"""
    precision(C::BinaryConfusionMatrix)

Return the ratio of positive samples in all samples classified as positive, i.e

```math
\mathrm{precision} = \frac{tp}{tp + fp}
```

Can be also called via alias `positive_predictive_value`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> precision(y, ŷ)
0.6

julia> positive_predictive_value(y, ŷ)
0.6
```
"""
Base.precision(args...) = apply(Precision(), args...)
const positive_predictive_value = precision

struct NegativePredictiveValue <: AbstractMetric end
function apply(::NegativePredictiveValue, C::BinaryConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = tn / (tn + fn)
    return isnan(val) ? one(val) : val
end

function apply(::NegativePredictiveValue, C::ConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = tn ./ (tn .+ fn)
    val[isnan.(val)] .= 1
    return val
end

@doc raw"""
    negative_predictive_value(C::BinaryConfusionMatrix)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{negative\_predictive\_value} = \frac{tn}{tn + fn}
```

Can be also called via alias `positive_predictive_value`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_predictive_value(y, ŷ)
0.4
```
"""
negative_predictive_value(args...) = apply(NegativePredictiveValue(), args...)

struct FalseDiscoveryRate <: AbstractMetric end
function apply(::FalseDiscoveryRate, C::BinaryConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = fp / (tp + fp)
    return isnan(val) ? zero(val) : val
end

function apply(::FalseDiscoveryRate, C::ConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = fp ./ (tp .+ fp)
    val[isnan.(val)] .= 1
    return val
end

@doc raw"""
    false_discovery_rate(C::BinaryConfusionMatrix)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{false\_discovery\_rate} = \frac{fp}{fp + tp}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_discovery_rate(y, ŷ)
0.4
```
"""
false_discovery_rate(args...) = apply(FalseDiscoveryRate(), args...)

struct FalseOmissionRate <: AbstractMetric end
function apply(::FalseOmissionRate, C::BinaryConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = fn / (tn + fn)
    return isnan(val) ? one(val) : val
end

function apply(::FalseOmissionRate, C::ConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = fn ./ (tn .+ fn)
    val[isnan.(val)] .= 1
    return val
end

@doc raw"""
    false_omission_rate(C::BinaryConfusionMatrix)

Return the ratio of positive samples in all samples classified as negatives, i.e

```math
\mathrm{false\_omission\_rate} = \frac{fn}{fn + tn}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_omission_rate(y, ŷ)
0.6
```
"""
false_omission_rate(args...) = apply(FalseOmissionRate(), args...)

struct ThreatScore <: AbstractMetric end
function apply(::ThreatScore, C::AbstractConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)
    fn = false_negatives(C)

    return tp ./ (tp .+ fn .+ fp)
end

@doc raw"""
    threat_score(C::BinaryConfusionMatrix)

Return threat score defined as

```math
\mathrm{threat\_score} = \frac{tp}{tp + fn + fp}
```

Can be also called via alias `critical_success_index`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> threat_score(y, ŷ)
0.375

julia> critical_success_index(y, ŷ)
0.375
```
"""
threat_score(args...) = apply(ThreatScore(), args...)
const critical_success_index = threat_score

struct FβScore{T<:Real} <: AbstractMetric
    β::T
end
function apply(m::FβScore, C::AbstractConfusionMatrix)
    prec = precision(C)
    rec = recall(C)
    β = m.β

    return @. (1 + β^2) * prec * rec / (β^2 * prec + rec)
end

@doc raw"""
    fβ_score(args...)

Return fβ score defined as

```math
\mathrm{fβ\_score} = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> fβ_score(y, ŷ)
0.5454545454545454

julia> fβ_score(y, ŷ; β = 2)
0.5172413793103449
```
"""
fβ_score(args...; β::Real=1) = apply(ThreatScore(β), args...)

struct MatthewsCorrelationCoefficient <: AbstractMetric end
function apply(::MatthewsCorrelationCoefficient, C::AbstractConfusionMatrix)
    tp = true_positives(C)
    tn = true_negatives(C)
    fp = false_positives(C)
    fn = false_negatives(C)

    return @. (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
end

@doc raw"""
    matthews_correlation_coefficient(C::BinaryConfusionMatrix)

Return matthews correlation coefficient defined as

```math
\mathrm{matthews\_correlation\_coefficient} = \frac{tp \cdot tn - fp \cdot fn}{\sqrt{(tp + fp)(tp + fn)(tn + fp)(tn + fn)}}
```

Can be also called via alias `mcc`.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> matthews_correlation_coefficient(y, ŷ)
0.0

julia> mcc(y, ŷ)
0.0
```
"""
matthews_correlation_coefficient(args...) = apply(MatthewsCorrelationCoefficient(), args...)
const mcc = matthews_correlation_coefficient

struct Quant <: AbstractMetric end
function apply(::Quant, C::AbstractConfusionMatrix)
    p = positives(C)
    n = negatives(C)
    tn = true_negatives(C)
    fn = false_negatives(C)

    return @. (fn + tn) / (p + n)
end

@doc raw"""
    quant(C::BinaryConfusionMatrix)

Return estimate of the quantile on classification scores that was used as a decision threshold

```math
\mathrm{quant} = \frac{fn + tn}{p + n}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> q = quant(y, ŷ)
0.5

julia> quant(y, scores, quantile(scores, q))
0.5
```
"""
quant(args...) = apply(Quant(), args...)

struct TopQuant <: AbstractMetric end
apply(::TopQuant, C::AbstractConfusionMatrix) = 1 .- quant(C)

@doc raw"""
    topquant(C::BinaryConfusionMatrix)

Return estimate of the top-quantile on classification scores that was used as a decision threshold

```math
\mathrm{topquant} = 1 - \mathrm{quant}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> scores = [0.2, 0.7, 0.3, 0.6, 0.8, 0.4, 0.3, 0.5, 0.7, 0.9];

julia> q = topquant(y, ŷ)
0.5

julia> topquant(y, scores, quantile(scores, q))
0.5
```
"""
topquant(args...) = apply(TopQuant(), args...)

struct PositiveLikelihoodRatio <: AbstractMetric end
function apply(::PositiveLikelihoodRatio, C::AbstractConfusionMatrix)
    return true_positive_rate(C) ./ false_positive_rate(C)
end

@doc raw"""
    positive_likelihood_ratio(C::BinaryConfusionMatrix)

Return positive likelihood ratio defined as

```math
\mathrm{positive\_likelihood\_ratio} = \frac{tpr}{fpr}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> positive_likelihood_ratio(y, ŷ)
1.0
```
"""
positive_likelihood_ratio(args...) = apply(PositiveLikelihoodRatio(), args...)

struct NegativeLikelihoodRatio <: AbstractMetric end
function apply(::NegativeLikelihoodRatio, C::AbstractConfusionMatrix)
    return false_negative_rate(C) ./ true_negative_rate(C)
end

@doc raw"""
    negative_likelihood_ratio(C::BinaryConfusionMatrix)

Return negative likelihood ratio defined as

```math
\mathrm{negative\_likelihood\_ratio} = \frac{fnr}{tnr}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_likelihood_ratio(y, ŷ)
1.0
```
"""
negative_likelihood_ratio(args...) = apply(NegativeLikelihoodRatio(), args...)

struct DiagnosticOddsRatio <: AbstractMetric end
function apply(::DiagnosticOddsRatio, C::AbstractConfusionMatrix)
    tpr = true_positive_rate(C)
    tnr = true_negative_rate(C)
    fpr = false_positive_rate(C)
    fnr = false_negative_rate(C)

    return @. tpr * tnr / (fpr * fnr)
end

@doc raw"""
    diagnostic_odds_ratio(C::BinaryConfusionMatrix)

Return diagnostic odds ratio defined as

```math
\mathrm{diagnostic\_odds\_ratio} = \frac{tpr \cdot tnr}{fpr \cdot fnr}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> diagnostic_odds_ratio(y, ŷ)
1.0
```
"""
diagnostic_odds_ratio(args...) = apply(DiagnosticOddsRatio(), args...)

struct Prevalence <: AbstractMetric end
function apply(::Prevalence, C::AbstractConfusionMatrix)
    p = positives(C)
    n = negatives(C)

    return @. p / (p + n)
end

@doc raw"""
    prevalence(C::BinaryConfusionMatrix)

Return prevalence defined as

```math
\mathrm{prevalence} = \frac{p}{p + n}
```

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> prevalence(y, ŷ)
0.6
```
"""
prevalence(args...) = apply(Prevalence(), args...)

# ------------------------------------------------------------------------------------------
# multiclass classification metrics
# ------------------------------------------------------------------------------------------
struct Accuracy <: AbstractMetric end
apply(::Accuracy, C::AbstractConfusionMatrix) = sum(diag(C)) / sum(C)

@doc raw"""
    accuracy(C::ConfusionMatrix)

Return accuracy defined as

```math
\mathrm{accuracy} = \frac{tp + tn}{p + n}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> accuracy(y, ŷ)
0.5
```
"""
accuracy(args...) = apply(Accuracy(), args...)

struct BalancedAccuracy <: AbstractMetric end
function apply(::BalancedAccuracy, C::AbstractConfusionMatrix)
    return mean(diag(C) / sum(C; dims=2))
end

@doc raw"""
    balanced_accuracy(C::ConfusionMatrix)

Return balanced accuracy defined as

```math
\mathrm{balanced\_accuracy} = \frac{1}{2}(tpr + tnr)
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_accuracy(y, ŷ)
0.5
```
"""
balanced_accuracy(args...) = apply(BalancedAccuracy(), args...)

struct ErrorRate <: AbstractMetric end
apply(::ErrorRate, C::AbstractConfusionMatrix) = 1 - accuracy(x)

@doc raw"""
    error_rate(C::ConfusionMatrix)

Return error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{accuracy}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> error_rate(y, ŷ)
0.5
```
"""
error_rate(args...) = apply(ErrorRate(), args...)

struct BalancedErrorRate <: AbstractMetric end
apply(::BalancedErrorRate, C::AbstractConfusionMatrix) = 1 - balanced_accuracy(x)

@doc raw"""
    balanced_error_rate(C::ConfusionMatrix)

Return balanced error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{balanced\_accuracy}
```

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_error_rate(y, ŷ)
0.5
```
"""
balanced_error_rate(args...) = apply(BalancedErrorRate(), args...)
