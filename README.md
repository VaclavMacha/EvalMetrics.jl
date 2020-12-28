
[![Build Status](https://github.com/VaclavMacha/EvalMetrics.jl/workflows/CI/badge.svg)](https://github.com/VaclavMacha/EvalMetrics.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/VaclavMacha/EvalMetrics.jl/badge.svg?branch=master)](https://coveralls.io/github/VaclavMacha/EvalMetrics.jl?branch=master)
[![codecov.io](http://codecov.io/github/VaclavMacha/EvalMetrics.jl/coverage.svg?branch=master)](http://codecov.io/github/VaclavMacha/EvalMetrics.jl?branch=master)

# EvalMetrics.jl
Utility package for scoring binary classification models. 


## Installation
Execute the following command in Julia Pkg REPL (`EvalMetrics.jl` requires julia 1.0 or higher)
```julia
(v1.5) pkg> add EvalMetrics
```

## Usage 

### Quickstart
The fastest way of getting started is to use a simple `binary_eval_report` function in the following way:

```julia
julia> using EvalMetrics, Random

julia> Random.seed!(123);

julia> targets = rand(0:1, 100);

julia> scores = rand(100);

julia> binary_eval_report(targets, scores)
Dict{String,Real} with 8 entries:
  "precision@fpr0.05"          => 0.0
  "recall@fpr0.05"             => 0.0
  "accuracy@fpr0.05"           => 0.45
  "au_prcurve"                 => 0.460134
  "samples"                    => 100
  "true negative rate@fpr0.05" => 0.957447
  "au_roccurve"                => 0.42232
  "prevalence"                 => 0.53
  
julia> binary_eval_report(targets, scores, 0.001)
Dict{String,Real} with 8 entries:
  "recall@fpr0.001"             => 0.0
  "au_prcurve"                  => 0.460134
  "samples"                     => 100
  "precision@fpr0.001"          => 1.0
  "au_roccurve"                 => 0.42232
  "accuracy@fpr0.001"           => 0.47
  "prevalence"                  => 0.53
  "true negative rate@fpr0.001" => 1.0
```

### Confusion Matrix
The core the package is the `ConfusionMatrix` structure, which represents the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) in the following form
|                         | Actual positives       | Actual negatives       |
| ------------------------| :--------------------: | :--------------------: |
| **Predicted positives** | tp (# true positives)  | fp (# false positives) |
| **Predicted negatives** | fn (# false negatives) | tn (# true negatives)  |
|                         | p  (# positives)       | n (# negatives)        |

The confusion matrix can be calculated from targets and predicted values or from targets, scores, and one or more decision thresholds 
```julia
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
| `error_rate`                       |                                      |
| `balanced_error_rate`              |                                      |
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
Different label encodings are considered common in different machine learning applications. For example, support vector machines use `1` as a positive label and `-1` as a negative label. On the other hand, it is common for neural networks to use `0` as a negative label. The package provides some basic label encodings listed in the following table
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
- `threshold_at_tpr(targets::AbstractVector, scores::RealVector, tpr::Real)` returns the largest threshold `t` that satisfies `true_positive_rate(targets, scores, t) >= tpr`
- `threshold_at_tnr(targets::AbstractVector, scores::RealVector, tnr::Real)` returns the smallest threshold `t` that satisfies `true_negative_rate(targets, scores, t) >= tnr`
- `threshold_at_fpr(targets::AbstractVector, scores::RealVector, fpr::Real)` returns the smallest threshold `t` that satisfies `false_positive_rate(targets, scores, t) <= fpr`
- `threshold_at_fnr(targets::AbstractVector, scores::RealVector, fnr::Real)` returns the largest threshold `t` that satisfies `false_negative_rate(targets, scores, t) <= fnr`

All four functions can be called with an encoding of type `AbstractEncoding` as the first parameter to use a different encoding than default.

### Evaluation curves
Functionality for measuring performance with curves is implemented in the package as well. For example, a precision-recall (PR) curve can be computed as follows:
```julia
julia> scores = [0.74, 0.48, 0.23, 0.91, 0.33, 0.92, 0.83, 0.61, 0.68, 0.09];

julia> targets = collect(1:10 .>= 3);

julia> prcurve(targets, scores)
([1.0, 0.875, 0.75, 0.625, 0.625, 0.5, 0.375, 0.375, 0.25, 0.125, 0.0],
 [0.8, 0.7777777777777778, 0.75, 0.7142857142857143, 0.8333333333333334, 0.8, 0.75, 1.0, 1.0, 1.0, 1.0])

```

All possible calls:
- `prcurve(targets::AbstractVector, scores::RealVector)` returns all `length(target) + 1` points
- `prcurve(enc::AbstractEncoding, target::AbstractVector, scores::RealVector)` makes different encodings possible
- `prcurve(targets::AbstractVector, scores::RealVector, thres::RealVector)` uses provided threshols to compute individual points
- `prcurve(enc::AbstractEncoding, target::AbstractVector, scores::RealVector, thres::RealVector)` 
- `prcurve(cms::AbstractVector{<:ConfusionMatrix})`

We can also compute area under the curve using the `auc_trapezoidal` function which uses the trapezoidal rule as follows:
```julia
julia> auc_trapezoidal(prcurve(targets, scores)...)
0.8595734126984128
```

However, a convenience function `au_prcurve` is provided with exactly the same signature as `prcurve` function. Moreover, any `curve(PRCurve, args...)` or `auc(PRCurve, args...)` call is equivalent to `prcurve(args...)` and `au_prcurve(args...)`, respectively.

Besides PR curve, Receiver operating characteristic (ROC) curve is also available out of the box with analogical definitions of `roccurve` and `au_roccurve`.

All points of the curve, as well as area under curve scores are computed using the highest possible resolution by default. This can be changed by a keyword argument `npoints`
```julia
julia> length.(prcurve(targets, scores))
(11, 11)
julia> length.(prcurve(targets, scores; npoints=9))
(9, 9)
julia> auprcurve(targets, scores)
0.8595734126984128
julia> au_prcurve(targets, scores; npoints=9)
0.8826388888888889
```

#### Plotting
For plotting purposes, `EvalMetrics.jl` provides recipes for the `Plots` library:

```julia
julia> using Plots; pyplot()
julia> using Random, MLBase; Random.seed!(42);
julia> scores = sort(rand(10000));
julia> targets = scores .>= 0.99;
julia> targets[MLBase.sample(findall(0.98 .<= scores .< 0.99), 30; replace = false)] .= true;
julia> targets[MLBase.sample(findall(0.99 .<= scores .< 0.995), 30; replace = false)] .= false;
```

Then, any of the following can be used:

- `prplot(targets::AbstractVector, scores::RealVector)` to use the full resolution:

```julia
julia> prplot(targets, scores)
```
<p align="center">
  <img src="docs/pr1.png?raw=true">
</p>

- `prplot(targets::AbstractVector, scores::RealVector, thresholds::RealVector)` to specify thresholds that will be used
- `prplot!(enc::AbstractEncoding, targets::AbstractVector, scores::RealVector)` to use a different encoding than default
- `prplot!(enc::AbstractEncoding, targets::AbstractVector, scores::RealVector, thresholds::RealVector)`

Furthermore, one can use vectors of vectors like `[targets1, targets2]` and `[scores1, scores2])` to plot multiple curves at once. The calls stay the same:

```julia
julia> prplot([targets, targets], [scores, scores .+ rand(10000) ./ 5])
```
<p align="center">
  <img src="docs/pr2.png?raw=true">
</p>

For ROC curve use `rocplot` analogically:

```julia
julia> rocplot(targets, scores)
```
<p align="center">
  <img src="docs/roc1.png?raw=true">
</p>

```julia
julia> rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5])
```
<p align="center">
  <img src="docs/roc2.png?raw=true">
</p>

'Modifying' versions with exclamation marks `prplot!` and `rocplot!` work as well. 

The appearance of the plot can be changed in exactly the same way as with `Plots` library. Therefore, keyword arguments such as `xguide`, `xlims`, `grid`, `fill` can all be used:

```julia
julia> prplot(targets, scores; xguide="RECALL", fill=:green, grid=false, xlims=(0.8, 1.0))
```
<p align="center">
  <img src="docs/pr3.png?raw=true">
</p>

```julia
julia> rocplot(targets, scores, title="Title", label="experiment", xscale=:log10)
```
<p align="center">
  <img src="docs/roc3.png?raw=true">
</p>

Here, limits on x axis are appropriately changed, unless overridden by using `xlims` keyword argument.

```julia
julia> rocplot([targets, targets], [scores, scores .+ rand(10000) ./ 5], label=["a" "b";])
```
<p align="center">
  <img src="docs/roc4.png?raw=true">
</p>

By default, plotted curves have 300 points, which are sampled to retain as much information as possible. This amounts to sampling false positive rate in case of ROC curves and true positive rate in case of PR curves instead of raw thresholds. The number of points can be again changed by keyword argument `npoints`:

```julia
julia> prplot(targets, scores; npoints=Inf, label="Original") 
julia> prplot!(targets, scores; npoints=10, label="Sampled (10 points)") 
julia> prplot!(targets, scores; npoints=100, label="Sampled (100 points)") 
julia> prplot!(targets, scores; npoints=1000, label="Sampled (1000 points)") 
julia> prplot!(targets, scores; npoints=5000, label="Sampled (5000 points)") 
```
<p align="center">
  <img src="docs/pr4.png?raw=true">
</p>

Note that even though we visuallize smaller number of points, the displayed auc score is computed from all points. In case when logarithmic scale is used, the sampling is also done in logarithmic scale.

Other than that, `diagonal` keyword indicates the diagonal in the plot, and `aucshow` toggles, whether auc score is appended to a label:
```julia
julia> rocplot(targets, scores; aucshow=false, label="a", diagonal=true)
```
<p align="center">
  <img src="docs/roc5.png?raw=true">
</p>

#### User-defined curves

PR and ROC curves are available out of the box. Additional curve definitions can be provided in the similar way as new metrics are defined using macro `@curve` and defining `apply` function, which computes a point on the curve. For instance, ROC curve can be defined this way:

```julia
julia> import EvalMetrics: @curve, apply 

julia> @curve MyROCCurve

julia> apply(::Type{MyROCCurve}, cms::AbstractVector{ConfusionMatrix{T}}) where T <: Real =
    (false_positive_rate(cms), true_positive_rate(cms))

julia> myroccurve(targets, scores) == roccurve(targets, scores)
true
```

In order to be able to sample from x axis while plotting, `sampling_function` and `lowest_metric_value` must be provided as well.
