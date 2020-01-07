# EvalMetrics.jl


## Installation


## Usage


### Confusion matrix 


|                     | Actual positives   | Actual negatives    |
| -----------------   | :-----:            | :-----:             |
| Predicted positives | tp (true positive) | fp (fasle positive) |
| Predicted negatives | fn (true positive) | tp (true negative)  |


```julia
target  = [  1,   1,   1,   1,   0,   0,   0,   0,   0,   0]
predict = [  1,   1,   0,   0,   1,   1,   0,   0,   0,   0]
scores  = [0.7, 0.8, 0.3, 0.2, 0.8, 0.9, 0.2, 0.1, 0.2, 0.3]
t = 0.4

c1 = counts(target, predict)
c2 = counts(target, scores, t)
```

```julia
Counts{Int64}(p = 4, n = 6, tp = 2, tn = 4, fp = 2, fn = 2)
```


### Classification metrics 


| Classification metric              | Aliases                              |
| -----------------                  | :-----:                              |
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


```julia
precision(c1)
precision(target, predict)
precision(target, scores, t)
```

```julia
0.5
```


#### Defining own classification metric using `@usermetric`

```julia
import EvalMetrics: @usermetric

@usermetric myfunc(x::Counts) = x.tp/(x.tp + x.fp)
```

```julia
myfunc (generic function with 3 methods)
```

 - `precision(x::Counts)`
 - `precision(target::IntegerVector predict::IntegerVector)`
 - `precision(target::IntegerVector, scores::RealVector, threshold::Real)`

### Threshold functions

### Curve function

## Todos
 - Write MORE Tests

## License
...

## Credits
...