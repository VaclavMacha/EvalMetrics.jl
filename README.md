
[![Build Status](https://travis-ci.com/VaclavMacha/EvalMetrics.jl.svg?branch=master)](https://travis-ci.com/VaclavMacha/EvalMetrics.jl)
[![Coverage Status](https://coveralls.io/repos/github/VaclavMacha/EvalMetrics.jl/badge.svg?branch=master)](https://coveralls.io/github/VaclavMacha/EvalMetrics.jl?branch=master)
[![codecov.io](http://codecov.io/github/VaclavMacha/EvalMetrics.jl/coverage.svg?branch=master)](http://codecov.io/github/VaclavMacha/EvalMetrics.jl?branch=master)

# EvalMetrics.jl
Utility package for scoring binary classification models. 


## Installation
Execute the following command in Julia Pkg REPL
```julia
(v1.4) pkg> add EvalMetrics
```

## Usage
The core the package is the `ConfusionMatrix` structure, which represents the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) in the following form
|                         | Actual positives       | Actual negatives       |
| ------------------------| :--------------------: | :--------------------: |
| **Predicted positives** | tp (# true positives)  | fp (# false positives) |
| **Predicted negatives** | fn (# false negatives) | tn (# true negatives)  |
|                         | p  (# positives)       | n (# negatives)        |

The confusion matrix can be calculated from targets and predicted values or from targets, scores, and one or more decision thresholds 
```julia
julia> using EvalMetrics, Random

julia> Random.seed!(123);

julia> targets = rand(0:1, 100);

julia> scores = rand(100);

julia> thres = 0.6;

julia> predicts  = scores .>= thres;

julia> cm1 = ConfusionMatrix(targets, predicts)
ConfusionMatrix{Int64}(53, 47, 18, 24, 23, 35)

julia> cm2 = ConfusionMatrix(targets, scores, thres)
ConfusionMatrix{Int64}(53, 47, 18, 24, 23, 35)

julia> cm3 = ConfusionMatrix(targets, scores, thres)
ConfusionMatrix{Int64}(53, 47, 18, 24, 23, 35)

julia> cm4 = ConfusionMatrix(targets, scores, [thres, thres])
2-element Array{ConfusionMatrix{Int64},1}:
 ConfusionMatrix{Int64}(53, 47, 18, 24, 23, 35)
 ConfusionMatrix{Int64}(53, 47, 18, 24, 23, 35)
```
The package provides many basic classification metrics based on the confusion matrix.  The following table provides a list of all available metrics and its aliases
| Classification metric              | Aliases                              |
| ---------------------------------- | :----------------------------------: |
| `true_positive`                    |                                      |
| `true_negative`                    |                                      |
| `false_positive`                   |                                      |
| `false_negative`                   |                                      |
| `true_positive_rate`               | `sensitivity`,  `recall`, `hit_rate` |
| `true_negative_rate`               | `specificity`,  `selectivity`        |
| `false_positive_rate`              | `fall_out`, `type_I_error`           |
| `false_negative_rate`              | `miss_rate`, `type_II_error`         |
| `precision`                        | `positive_predictive_value`          |
| `negative_predictive_value`        |                                      |
| `false_discovery_rate`             |                                      |
| `false_omission_rate`              |                                      |
| `threat_score`                     | `critical_success_index`             |
| `accuracy`                         |                                      |
| `balanced_accuracy`                |                                      |
| `f1_score`                         |                                      |
| `fβ_score`                         |                                      |
| `matthews_correlation_coefficient` | `mcc`                                |
| `quant`                            |                                      |
| `positive_likelihood_ratio`        |                                      |
| `negative_likelihood_ratio`        |                                      |
| `diagnostic_odds_ratio`            |                                      |
| `prevalence`                       |                                      |

Each metric can be computed from the `ConfusionMatrix` structure 
```julia
julia> recall(cm1)
0.33962264150943394

julia> recall(cm2)
0.33962264150943394

julia> recall(cm3)
0.33962264150943394

julia> recall(cm4)
2-element Array{Float64,1}:
 0.33962264150943394
 0.33962264150943394
```
The other option is to compute the metric directly from targets and predicted values or from targets, scores, and one or more decision thresholds
```julia
julia> recall(targets, predicts)
0.33962264150943394

julia> recall(targets, scores, thres)
0.33962264150943394

julia> recall(targets, scores, thres)
0.33962264150943394

julia> recall(targets, scores, [thres, thres])
2-element Array{Float64,1}:
 0.33962264150943394
 0.33962264150943394
```

### User defined classification metrics
It may occur that some useful metric is not defined in the package. To simplify the process of defining a new metric, the package provides the `@metric` macro and `apply` function. 
```julia
import EvalMetrics: @metric, metric

@metric MyRecall

apply(::Type{MyRecall}, x::ConfusionMatrix) = x.tp/x.p
```
In the previous example, macro `@metric` defines a new abstract type `MyRecall` (used for dispatch) and a function `myrecall` (for easy use of the new metric).  With defined abstract type `MyRecall`, the next step is to define a new method for the `apply` function. This method must have exactly two input arguments: `Type{MyRecall}` and `ConfusionMatrix`.  If another argument is needed, it can be added as a keyword argument.
```julia
apply(::Type{Fβ_score}, x::ConfusionMatrix; β::Real = 1) =
    (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))
```
It is easy to check that the `myrecall` metric returns the same outputs as the `recall` metric defined in the package
```julia
julia> myrecall(cm1)
0.33962264150943394

julia> myrecall(cm2)
0.33962264150943394

julia> myrecall(cm3)
0.33962264150943394

julia> myrecall(cm4)
2-element Array{Float64,1}:
 0.33962264150943394
 0.33962264150943394
julia> myrecall(targets, predicts)
0.33962264150943394

julia> myrecall(targets, scores, thres)
0.33962264150943394

julia> myrecall(targets, scores, thres)
0.33962264150943394

julia> myrecall(targets, scores, [thres, thres])
2-element Array{Float64,1}:
 0.33962264150943394
 0.33962264150943394
```

### Label encodings
Different label encodings are considered common in different machine learning applications. For example, supporting vector machines use `1` as a positive label and `-1` as a negative label. On the other hand, it is common for neural networks to use `0` as a negative label. The package provides some basic label encodings listed in the following table
| Encoding                                               | positive label(s) | negative label(s) |
| ------------------------------------------------------ | :---------------: | :---------------: |
| `OneZero(::Type{T})`                                   | `one(T)`          | `zero(T)`         |
| `OneMinusOne(::Type{T})`                               | `one(T)`          | `-one(T)`         |
| `OneTwo(::Type{T})`                                    | `one(T)`          | `2*one(T)`        |
| `OneVsOne(::Type{T}, pos::T, neg::T)`                  | `pos`             | `neg`             |
| `OneVsRest(::Type{T}, pos::T, neg::AbstractVector{T})` | `pos`             | `neg`             |
| `RestVsOne(::Type{T}, pos::AbstractVector{T}, neg::T)` | `pos`             | `neg`             |

The `current_encoding` function can be used to verify which encoding is currently in use (by default it is `OneZero` encoding)
```julia
julia> enc = current_encoding()
OneZero{Float64}:
   positive class: 1.0
   negative class: 0.0
```
One way to use a different encoding is to pass the new encoding as the first argument
```julia
julia> enc_new = OneVsOne(:positive, :negative)
OneVsOne{Symbol}:
   positive class: positive
   negative class: negative

julia> targets_recoded = recode.(enc, enc_new, targets);

julia> predicts_recoded = recode.(enc, enc_new, predicts);

julia> recall(enc, targets, predicts)
0.33962264150943394

julia> recall(enc_new, targets_recoded, predicts_recoded)
0.33962264150943394
```
The second way is to change the current encoding to the one you want
```julia
julia> set_encoding(OneVsOne(:positive, :negative))
OneVsOne{Symbol}:
   positive class: positive
   negative class: negative

julia> recall(targets_recoded, predicts_recoded)
0.33962264150943394
```

### Decision thresholds for classification
The package provides a `thresholds(scores::RealVector, n::Int)` , which returns `n` decision thresholds which correspond to `n` evenly spaced quantiles of the given `scores` vector. The default value of `n` is `length(scores) + 1`.  The `thresholds` function has two keyword arguments `reduced::Bool` and `zerorecall::Bool`
- If `reduced` is `true` (default), then the function returns `min(length(scores) + 1, n)` thresholds.
- If `zerorecall`  is `true` (default), then the largest threshold is `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.

The package also provides some other useful utilities
- `threshold_at_tpr(target::IntegerVector, scores::RealVector, tpr::Real)` returns the largest threshold `t` that satisfies `true_positive_rate(target, scores, t) >= tpr`
- `threshold_at_tnr(target::IntegerVector, scores::RealVector, tnr::Real)`returns the smallest threshold `t` that satisfies `true_negative_rate(target, scores, t) >= tnr`
- `threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)` returns the smallest threshold `t` that satisfies `false_positive_rate(target, scores, t) <= fpr`
- `threshold_at_fnr(target::IntegerVector, scores::RealVector, fnr::Real)` returns the largest threshold `t` that satisfies `false_negative_rate(target, scores, t) <= fnr`
