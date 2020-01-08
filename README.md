# EvalMetrics.jl
Utility package for scoring binary classification models. 


## Installation
The package is not registered.
```julia
(1.2v) pkg> add https://github.com/VaclavMacha/EvalMetrics.jl
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
Fnuction `counts` has three methods
- `counts(target::IntegerVector, predict::IntegerVector)`  returns an `Counts` instance based on given vector `target` of target labels and `predict` of predicted labels. 
- `counts(target::IntegerVector, scores::RealVector, thres::Real)`  returns an `Counts` instance based on given vector `target` of target labels  and predicted labels `predict[i] = scores[i] >= thres` computed from the given vector `scores` of classification scores and decision threshold `thres`. 
- `counts(target::IntegerVector, scores::RealVector, thres::RealVector)` returns a vector of `Counts` instances. This vector is of the type `Array{Counts{T},1}`, but the alias `CountsVector{T}` is often used in the code for simplicity. 

```julia
julia> using EvalMetrics

julia> target  = [  1,   1,   1,   1,   0,   0,   0,   0,   0,   0];

julia> predict = [  1,   1,   0,   0,   1,   1,   0,   0,   0,   0];

julia> scores  = [0.7, 0.8, 0.3, 0.2, 0.8, 0.9, 0.2, 0.1, 0.2, 0.3];

julia> thres   = 0.4;

julia> counts(target, predict)
Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)

julia> counts(target, scores, thres)
Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)

julia> counts(target, scores, [thres, thres])
2-element Array{Counts{Int64},1}:
 Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)
 Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)
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
- `precision(target::IntegerVector, predict::IntegerVector)`  is equivalet to `precision(counts(target, predict))`.
- `precision(target::IntegerVector, scores::RealVector, thres::Real)`  is equivalet to `precision(counts(target, scores, thres))`.
- `precision(x::VectorCounts)`  is equivalet to `precision.(x)`.
- `precision(target::IntegerVector, scores::RealVector, thres::RealVector)`  is equivalet to `precision.(counts(target, scores, thres))`.

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
(the first argument of the core function must be of type `Counts`)  and all the remaining methods for this function are generated automatically. The following example shows how `precision` is defined
```julia
julia> import EvalMetrics: @usermetric

julia> @usermetric my_precision(x::Counts) = x.tp/(x.tp + x.fp)
my_precision (generic function with 5 methods)
```

Note that the input to the `@usermetric` must be a valid function definition. The following will not work
```julia
julia> f(x::Counts) = 1
f (generic function with 1 method)

julia> @usermetric f
ERROR: LoadError: MethodError: no method matching @usermetric(::LineNumberNode, ::Module, ::Symbol)
```

The same input arguments as we used for the `precision` function from the package in the example above yields to the same results
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


####   :warning: **Avoid using `Array`, `Tuple`, etc. as arguments** 
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

### Threshold functions

### Other utilities
