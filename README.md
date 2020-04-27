[![Build Status](https://travis-ci.com/VaclavMacha/EvalMetrics.jl.svg?branch=master)](https://travis-ci.com/VaclavMacha/EvalMetrics.jl)
[![Coverage Status](https://coveralls.io/repos/github/VaclavMacha/EvalMetrics.jl/badge.svg?branch=master)](https://coveralls.io/github/VaclavMacha/EvalMetrics.jl?branch=master)
[![codecov.io](http://codecov.io/github/VaclavMacha/EvalMetrics.jl/coverage.svg?branch=master)](http://codecov.io/github/VaclavMacha/EvalMetrics.jl?branch=master)

# EvalMetrics.jl
Utility package for scoring binary classification models. 


## Installation
The package is not registered yet.
```julia
(v1.4) pkg> add https://github.com/VaclavMacha/EvalMetrics.jl
```

## Usage

The core function of the package is the `counts` function, which computes the values of the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

|                         | Actual positives       | Actual negatives       |
| ------------------------| :--------------------: | :--------------------: |
| **Predicted positives** | tp (# true positives)  | fp (# false positives) |
| **Predicted negatives** | fn (# false negatives) | tn (# true negatives)  |
|                         | p  (# positives)       | n (# negatives)        |

and returns them in the `Counts` structure
```julia
struct Counts{T<:Real}
    p::T
    n::T
    tp::T
    tn::T
    fp::T
    fn::T
end
```
Function `counts` has three methods
- `counts(target::IntegerVector, predict::IntegerVector)`  returns an `Counts` instance based on given vector `target` of target labels and `predict` of predicted labels. 
- `counts(target::IntegerVector, scores::RealVector, thres::Real)`  returns an `Counts` instance based on given vector `target` of target labels  and predicted labels `predict[i] = scores[i] >= thres` computed from the given vector `scores` of classification scores and decision threshold `thres`. 
- `counts(target::IntegerVector, scores::RealVector, thres::RealVector)` returns a vector of `Counts` instances. 
```julia
julia> using EvalMetrics

julia> target = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0];

julia> predict = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0];

julia> scores = [0.7, 0.8, 0.3, 0.2, 0.8, 0.9, 0.2, 0.1, 0.2, 0.3];

julia> thres = 0.4;

julia> counts(target, predict)
Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)

julia> counts(target, scores, thres)
Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)

julia> counts(target, scores, [thres, thres])
2-element Array{Counts{Int64},1}:
 Counts{Int64}
 Counts{Int64}
```

### Classification metrics 
The package provides many basic binary classification metrics based on the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).  The following table provides a list of all available metrics and its aliases

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

For ease of use, each metric has five methods. For example, assume `precision`
- `precision(x::Counts)`.
- `precision(target::IntegerVector, predict::IntegerVector)`  is equivalent to `precision(counts(target, predict))`.
- `precision(target::IntegerVector, scores::RealVector, thres::Real)`  is equivalent to `precision(counts(target, scores, thres))`.
- `precision(x::Vector{Counts})`  is equivalent to `precision.(x)`.
- `precision(target::IntegerVector, scores::RealVector, thres::RealVector)`  is equivalent to `precision.(counts(target, scores, thres))`.

The following example shows the expected behavior
```julia
julia> c = counts(target, predict);

julia> precision(c)
0.5

julia> precision(target, predict)
0.5

julia> precision(target, scores, thres)
0.5

julia> precision([c, c])
2-element Array{Float64,1}:
 0.5
 0.5
 
julia> precision(target, scores, [thres, thres])
2-element Array{Float64,1}:
 0.5
 0.5
```

### Defining own classification metric
It may occur that some useful metric is not defined in the package. To simplify the process of defining a new metric, the package provides the `@usermetric` macro. If this macro is used, the user has to define only the core function (the first argument of this is function must be of type `Counts`) such as
```julia
my_metric(x::Counts, args...; kwargs...) = ...
```
or 
```julia
function my_metric(x::Counts, args...; kwargs...)
    ...
end
```
(the first argument of the core function must be of type `Counts`)  and all the remaining methods for this function are generated automatically. Note that the input to the `@usermetric` must be a valid function definition. The following will not work
```julia
julia> f(x::Counts) = 1
f (generic function with 1 method)

julia> @usermetric f
ERROR: LoadError: MethodError: no method matching @usermetric(::LineNumberNode, ::Module, ::Symbol)
```
The following example shows how `precision` is defined
```julia
julia> import EvalMetrics: @usermetric

julia> @usermetric my_precision(x::Counts) = x.tp/(x.tp + x.fp)
my_precision (generic function with 5 methods)
```
The same input arguments as we used for the `precision` function from the package in the example above yields the same results
```julia
julia> my_precision(c)
0.5

julia> my_precision(target, predict)
0.5

julia> my_precision(target, scores, thres)
0.5

julia> my_precision([c, c])
2-element Array{Float64,1}:
 0.5
 0.5

julia> my_precision(target, scores, [thres, thres])
2-element Array{Float64,1}:
 0.5
 0.5
```


####   :warning: **Avoid using iterable objects as arguments** 
Since `@usermetric` uses [dot syntax](https://docs.julialang.org/en/v1/manual/functions/#man-vectorized-1) to define some methods, it is not recommended to use `Array`, `Tuple` and other iterable objects as arguments.  Such arguments may lead to potentially unwanted behavior
```julia
julia> @usermetric my_metric(x::Counts, y::Array) = y
my_metric (generic function with 5 methods)

julia> my_metric([c], [1,2,3])
ERROR: MethodError: no method matching my_metric(::Counts{Int64}, ::Int64)

julia> my_metric([c, c], [1,2,3])
ERROR: DimensionMismatch("arrays could not be broadcast to a common size")

julia> my_metric([c, c], [[1,2,3]])
2-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [1, 2, 3]
```
This can be fixed by defining such arguments as keyword arguments
```julia
julia> @usermetric my_metric_kwargs(x::Counts; y::Array = []) = y
my_metric_kwargs (generic function with 5 methods)

julia> my_metric_kwargs([c]; y = [1,2,3])
1-element Array{Array{Int64,1},1}:
 [1, 2, 3]

julia> my_metric_kwargs([c, c], y = [1,2,3])
2-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [1, 2, 3]

julia> my_metric_kwargs([c, c], y = [[1,2,3]])
2-element Array{Array{Array{Int64,1},1},1}:
 [[1, 2, 3]]
 [[1, 2, 3]]
```

### Thresholds function
The package provides a `thresholds(scores::RealVector, n::Int)` , which returns `n` decision thresholds which correspond to `n` evenly spaced quantiles of the given `scores` vector. The default value of `n` is `length(scores) + 1`.  The `thresholds` function has two keyword arguments `reduced::Bool` and `zerorecall::Bool`
- If `reduced` is `true` (default), then the function returns `min(length(scores) + 1, n)` thresholds.
- If `zerorecall`  is `true` (default), then the largest threshold is `maximum(scores)*(1 + eps())` otherwise `maximum(scores)`.
```julia
julia> target = 1:6 .>= 3;

julia> scores = 1:6;

julia> t1 = thresholds(scores);

julia> t2 = thresholds(scores, 8; reduced = true,  zerorecall = true);

julia> t3 = thresholds(scores, 8; reduced = true,  zerorecall = false);

julia> t4 = thresholds(scores, 8; reduced = false, zerorecall = true);

julia> t5 = thresholds(scores, 8; reduced = false, zerorecall = false);

julia> hcat(t1, t2, t3,
            recall(target, scores, t1),
            recall(target, scores, t2),
            recall(target, scores, t3))
7×6 Array{Float64,2}:
 1.0  1.0  1.0      1.0   1.0   1.0 
 2.0  2.0  1.83333  1.0   1.0   1.0 
 3.0  3.0  2.66667  1.0   1.0   1.0 
 4.0  4.0  3.5      0.75  0.75  0.75
 5.0  5.0  4.33333  0.5   0.5   0.5 
 6.0  6.0  5.16667  0.25  0.25  0.25
 6.0  6.0  6.0      0.0   0.0   0.25

julia> hcat(t4, t5,
            recall(target, scores, t4),
            recall(target, scores, t5))
8×4 Array{Float64,2}:
 1.0      1.0      1.0   1.0 
 1.83333  1.71429  1.0   1.0 
 2.66667  2.42857  1.0   1.0 
 3.5      3.14286  0.75  0.75
 4.33333  3.85714  0.5   0.75
 5.16667  4.57143  0.25  0.5 
 6.0      5.28571  0.25  0.25
 6.0      6.0      0.0   0.25
```

### Other utilities
The package also provides some other useful utilities
- `threshold_at_tpr(target::IntegerVector, scores::RealVector, tpr::Real)` returns the largest threshold `t` that satisfies `true_positive_rate(target, scores, t) >= tpr`
- `threshold_at_tnr(target::IntegerVector, scores::RealVector, tnr::Real)`returns the smallest threshold `t` that satisfies `true_negative_rate(target, scores, t) >= tnr`
- `threshold_at_fpr(target::IntegerVector, scores::RealVector, fpr::Real)` returns the smallest threshold `t` that satisfies `false_positive_rate(target, scores, t) <= fpr`
- `threshold_at_fnr(target::IntegerVector, scores::RealVector, fnr::Real)` returns the largest threshold `t` that satisfies `false_negative_rate(target, scores, t) <= fnr`
```julia
julia> using Test, Random

julia> Random.seed!(1234);

julia> target = rand(0:1, 10000);

julia> scores = rand(10000);

julia> rate   = 0.2;

julia> t1 = threshold_at_tpr(target, scores, rate);

julia> tpr = true_positive_rate(target, scores, t1)
0.2000402495471926

julia> t2  = threshold_at_tnr(target, scores, rate);

julia> tnr = true_negative_rate(target, scores, t2)
0.20015901411250248

julia> t3  = threshold_at_fpr(target, scores, rate);

julia> fpr = false_positive_rate(target, scores, t3)
0.19996024647187438

julia> t4  = threshold_at_fnr(target, scores, rate);

julia> fnr = false_negative_rate(target, scores, t4)
0.19983900181122963

julia> @testset "test thresholds_at_" begin
           @test tpr >= rate > true_positive_rate(target, scores, t1  + eps())
           @test tnr >= rate > true_negative_rate(target, scores, t2  - eps())
           @test fpr <= rate < false_positive_rate(target, scores, t3 - eps())
           @test fnr <= rate < false_negative_rate(target, scores, t4 + eps())
       end;
Test Summary:       | Pass  Total
test thresholds_at_ |    4      4 
```
- `auc(x::RealVector, y::RealVector)` returns the area under the curve computed using the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).
```julia
julia> x = 0:0.1:1
0.0:0.1:1.0

julia> y = 0:0.1:1
0.0:0.1:1.0

julia> auc(x,y)
0.5 
```
