# EvalMetrics.jl


## Installation


## Usage


### Confusion matrix 


|                     | Actual positives   | Actual negatives    |
| -----------------   | :-----:            | :-----:             |
| Predicted positives | tp (true positive) | fp (fasle positive) |
| Predicted negatives | fn (true positive) | tp (true negative)  |


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

#### Defining own classification metric using `@usermetric`

### Threshold functions

### Curve function

## Todos
 - Write MORE Tests

## License
...

## Credits
...