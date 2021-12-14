abstract type AbstractMetric end

function apply(M::Type{<:AbstractMetric}, args...; kwargs...)
    return apply(M, ConfusionMatrix(args...); kwargs...)
end

function apply(M::Type{<:AbstractMetric}, x::AbstractArray{<:ConfusionMatrix}; kwargs...)
    return apply.(M, x; kwargs...)
end

function apply(M::Type{<:AbstractMetric}, C::AbstractConfusionMatrix; kwargs...)
    name_lw = lowercase(string(M.name.name))
    N = size(C, 2)
    type = N == 2 ? "binary" : "multi-class"
    error("$(name_lw) not defined for $(type) confusion matrix")
    return
end

"""
    @metric

Macro to simplify the definition of new classification metrics.

# Examples

Using of the macro in the following way

```julia
import EvalMetrics: @metric, apply
@metric True_negative_rate specificity selectivity

apply(::Type{True_negative_rate}, C::ConfusionMatrix) = x.tn/x.n
```

is equivalent to

```julia
import EvalMetrics: apply, AbstractMetric

abstract type True_negative_rate <: AbstractMetric end

true_negative_rate(args...; kwargs...) = apply(True_negative_rate, args...; kwargs...)

apply(::Type{True_negative_rate}, C::ConfusionMatrix) = x.tn/x.n

const specificity = true_negative_rate
const selectivity = true_negative_rate
```
"""
macro metric(name, aliases...)
    name_lw = Symbol(lowercase(string(name)))

    e = quote
        abstract type $(esc(name)) <: AbstractMetric end

        function $(esc(name_lw))(args...; kwargs...)
            apply($(esc(name)), args...; kwargs...)
        end

        export $(esc(name_lw))
    end
    for alias in aliases
        e_alias = quote
            const $(esc(alias)) = $(esc(name_lw))
            export $(esc(alias))
        end
        append!(e.args, e_alias.args)
    end
    return e
end

# ------------------------------------------------------------------------------------------
# binary classification metrics
# ------------------------------------------------------------------------------------------

@metric Negatives
apply(::Type{Negatives}, C::BinaryConfusionMatrix) = sum(C[1, :])

@metric True_Negatives
apply(::Type{True_Negatives}, C::BinaryConfusionMatrix) = C[2, 2]

"""
    true_negatives(args...)

Return the number of correctly classified negative samples. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_negatives(y, ŷ)
2
```
"""
true_negatives

@metric False_Positives
apply(::Type{False_Positives}, C::BinaryConfusionMatrix) = C[1, 2]

"""
    false_positives(args...)

Return the number of incorrectly classified negative samples. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_positives(y, ŷ)
2
```
"""
false_positives

@metric Positives
apply(::Type{Positives}, C::BinaryConfusionMatrix) = sum(C[2, :])


@metric True_Positives
apply(::Type{True_Positives}, C::BinaryConfusionMatrix) = C[2, 2]

"""
    true_positives(args...)

Return the number of correctly classified positive samples. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> true_positives(y, ŷ)
3
```
"""
true_positives

@metric False_Negatives
apply(::Type{False_Negatives}, C::BinaryConfusionMatrix) = C[2, 1]

"""
    false_negatives(args...)

Return the number of incorrectly classified positive samples. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_negatives(y, ŷ)
3
```
"""
false_negatives

@metric True_Positive_Rate sensitivity recall hit_rate
function apply(::Type{True_Positive_Rate}, C::BinaryConfusionMatrix)
    return true_positives(C) / positives(C)
end

@doc raw"""
    true_positive_rate(args...)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_positive\_rate} = \frac{tp}{p}
```

Can be also called via aliases `sensitivity`,  `recall`, `hit_rate`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
true_positive_rate

@metric True_Negative_Rate specificity selectivity
function apply(::Type{True_Negative_Rate}, C::BinaryConfusionMatrix)
    return true_negatives(C) / negatives(C)
end

@doc raw"""
    true_negative_rate(args...)

Return the proportion of correctly classified positive samples, i.e

```math
\mathrm{true\_negative\_rate} = \frac{tn}{n}
```

Can be also called via aliases `specificity`,  `selectivity`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
true_negative_rate

@metric False_Positive_Rate fall_out type_I_error
function apply(::Type{False_Positive_Rate}, C::BinaryConfusionMatrix)
    return false_positives(C) / negatives(C)
end

@doc raw"""
    false_positive_rate(args...)

Return the proportion of incorrectly classified negative samples, i.e

```math
\mathrm{false\_positive\_rate} = \frac{tn}{p}
```

Can be also called via aliases `fall_out`, `type_I_error`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
false_positive_rate

@metric False_Negative_Rate miss_rate type_II_error
function apply(::Type{False_Negative_Rate}, C::BinaryConfusionMatrix)
    return false_negatives(C) / positives(C)
end

@doc raw"""
    false_negative_rate(args...)

Return the proportion of incorrectly classified positive samples, i.e

```math
\mathrm{false\_negative\_rate} = \frac{tp}{n}
```

Can be also called via aliases `miss_rate`, `type_II_error`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
false_negative_rate

@metric Precision positive_predictive_value
function apply(::Type{Precision}, C::BinaryConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = tp / (tp + fp)
    return isnan(val) ? one(val) : val
end

@doc raw"""
    precision(args...)

Return the ratio of positive samples in all samples classified as positive, i.e

```math
\mathrm{precision} = \frac{tp}{tp + fp}
```

Can be also called via alias `positive_predictive_value`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
precision

@metric Negative_Predictive_Value
function apply(::Type{Negative_Predictive_Value}, C::BinaryConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = tn / (tn + fn)
    return isnan(val) ? one(val) : val
end

@doc raw"""
    negative_predictive_value(args...)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{negative\_predictive\_value} = \frac{tn}{tn + fn}
```

Can be also called via alias `positive_predictive_value`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_predictive_value(y, ŷ)
0.4
```
"""
negative_predictive_value

@metric False_Discovery_Rate
function apply(::Type{False_Discovery_Rate}, C::BinaryConfusionMatrix)
    tp = true_positives(C)
    fp = false_positives(C)

    val = fp / (tp + fp)
    return isnan(val) ? zero(val) : val
end

@doc raw"""
    false_discovery_rate(args...)

Return the ratio of negative samples in all samples classified as positive, i.e

```math
\mathrm{false\_discovery\_rate} = \frac{fp}{fp + tp}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_discovery_rate(y, ŷ)
0.4
```
"""
false_discovery_rate

@metric False_Omission_Rate
function apply(::Type{False_Omission_Rate}, C::BinaryConfusionMatrix)
    tn = true_negatives(C)
    fn = false_negatives(C)

    val = fn / (tn + fn)
    return isnan(val) ? one(val) : val
end

@doc raw"""
    false_omission_rate(args...)

Return the ratio of positive samples in all samples classified as negatives, i.e

```math
\mathrm{false\_omission\_rate} = \frac{fn}{fn + tn}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> false_omission_rate(y, ŷ)
0.6
```
"""
false_omission_rate

@metric Threat_Score critical_success_index
function apply(::Type{Threat_Score}, C::BinaryConfusionMatrix)
    tp = true_positives(C)

    return tp / (tp + false_negatives(C) + false_positives(C))
end

@doc raw"""
    threat_score(args...)

Return threat score defined as

```math
\mathrm{threat\_score} = \frac{tp}{tp + fn + fp}
```

Can be also called via alias `critical_success_index`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
threat_score

@metric Fβ_Score
function apply(::Type{Fβ_Score}, C::BinaryConfusionMatrix; β::Real = 1)
    prec = precision(C)
    rec = recall(C)

    return (1 + β^2) * prec * rec / (β^2 * prec + rec)
end

@doc raw"""
    fβ_score(args...; β = 1)

Return fβ score defined as

```math
\mathrm{fβ\_score} = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{\beta^2 \cdot \mathrm{precision} + \mathrm{recall}}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
fβ_score

@metric Matthews_Correlation_Coefficient mcc
function apply(::Type{Matthews_Correlation_Coefficient}, C::BinaryConfusionMatrix)
    tp = true_positives(C)
    tn = true_negatives(C)
    fp = false_positives(C)
    fn = false_negatives(C)

    return (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
end

@doc raw"""
    matthews_correlation_coefficient(args...)

Return matthews correlation coefficient defined as

```math
\mathrm{matthews\_correlation\_coefficient} = \frac{tp \cdot tn - fp \cdot fn}{\sqrt{(tp + fp)(tp + fn)(tn + fp)(tn + fn)}}
```

Can be also called via alias `mcc`. See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
matthews_correlation_coefficient

@metric Quant
function apply(::Type{Quant}, C::BinaryConfusionMatrix)
    p = positives(C)
    n = negatives(C)
    tn = true_negatives(C)
    fn = false_negatives(C)

    return (fn + tn) / (p + n)
end

@doc raw"""
    quant(args...)

Return estimate of the quantile on classification scores that was used as a decision threshold

```math
\mathrm{quant} = \frac{fn + tn}{p + n}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
quant

@metric TopQuant
apply(::Type{TopQuant}, C::BinaryConfusionMatrix) = 1 - quant(x)

@doc raw"""
    topquant(args...)

Return estimate of the top-quantile on classification scores that was used as a decision threshold

```math
\mathrm{topquant} = 1 - \mathrm{quant}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

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
topquant

@metric Positive_Likelihood_Ratio
function apply(::Type{Positive_Likelihood_Ratio}, C::BinaryConfusionMatrix)
    return true_positive_rate(C) / false_positive_rate(C)
end

@doc raw"""
    positive_likelihood_ratio(args...)

Return positive likelihood ratio defined as

```math
\mathrm{positive\_likelihood\_ratio} = \frac{tpr}{fpr}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> positive_likelihood_ratio(y, ŷ)
1.0
```
"""
positive_likelihood_ratio

@metric Negative_Likelihood_Ratio
function apply(::Type{Negative_Likelihood_Ratio}, C::BinaryConfusionMatrix)
    return false_negative_rate(C) / true_negative_rate(C)
end

@doc raw"""
    negative_likelihood_ratio(args...)

Return negative likelihood ratio defined as

```math
\mathrm{negative\_likelihood\_ratio} = \frac{fnr}{tnr}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> negative_likelihood_ratio(y, ŷ)
1.0
```
"""
negative_likelihood_ratio

@metric Diagnostic_Odds_Ratio
function apply(::Type{Diagnostic_Odds_Ratio}, C::BinaryConfusionMatrix)
    tpr = true_positive_rate(C)
    tnr = true_negative_rate(C)
    fpr = false_positive_rate(C)
    fnr = false_negative_rate(C)

    return tpr * tnr / (fpr * fnr)
end

@doc raw"""
    diagnostic_odds_ratio(args...)

Return diagnostic odds ratio defined as

```math
\mathrm{diagnostic\_odds\_ratio} = \frac{tpr \cdot tnr}{fpr \cdot fnr}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> diagnostic_odds_ratio(y, ŷ)
1.0
```
"""
diagnostic_odds_ratio

@metric Prevalence
function apply(::Type{Prevalence}, C::BinaryConfusionMatrix)
    p = positives(C)
    n = negatives(C)

    return p / (p + n)
end

@doc raw"""
    prevalence(args...)

Return prevalence defined as

```math
\mathrm{prevalence} = \frac{p}{p + n}
```

See [`BinaryConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> using Statistics: quantile

julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> prevalence(y, ŷ)
0.6
```
"""
prevalence

# ------------------------------------------------------------------------------------------
# multiclass classification metrics
# ------------------------------------------------------------------------------------------
@metric Accuracy
function apply(::Type{Accuracy}, C::ConfusionMatrix)
    return sum(diag(C)) / sum(C)
end

@doc raw"""
    accuracy(args...)

Return accuracy defined as

```math
\mathrm{accuracy} = \frac{tp + tn}{p + n}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> accuracy(y, ŷ)
0.5
```
"""
accuracy

@metric Balanced_Accuracy
function apply(::Type{Balanced_Accuracy}, C::ConfusionMatrix)
    return mean(diag(C) / sum(C; dims = 2))
end

@doc raw"""
    balanced_accuracy(args...)

Return balanced accuracy defined as

```math
\mathrm{balanced\_accuracy} = \frac{1}{2}(tpr + tnr)
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_accuracy(y, ŷ)
0.5
```
"""
balanced_accuracy

@metric Error_Rate
apply(::Type{Error_Rate}, C::ConfusionMatrix) = 1 - accuracy(x)

@doc raw"""
    error_rate(args...)

Return error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{accuracy}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> error_rate(y, ŷ)
0.5
```
"""
error_rate

@metric Balanced_Error_Rate
apply(::Type{Balanced_Error_Rate}, C::ConfusionMatrix) = 1 - balanced_accuracy(x)

@doc raw"""
    balanced_error_rate(args...)

Return balanced error rate defined as

```math
\mathrm{error\_rate} = 1 - \mathrm{balanced\_accuracy}
```

See [`ConfusionMatrix`](@ref) for more information on possible input arguments.

# Examples

```jldoctest
julia> y = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1];

julia> ŷ = [0, 1, 0, 1, 1, 0, 0, 0, 1, 1];

julia> balanced_error_rate(y, ŷ)
0.5
```
"""
balanced_error_rate