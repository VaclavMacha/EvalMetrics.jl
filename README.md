# EvalMetrics.jl


## Installation


## Usage


### Confusion matrix 


|                     | Actual positives   | Actual negatives    |
| -----------------   | :-----:            | :-----:             |
| Predicted positives | tp (true positive) | fp (fasle positive) |
| Predicted negatives | fn (true positive) | tp (true negative)  |


### Classification metrics 


| Classification metric              | Formula                                                       | Aliases                              |
| -----------------                  | :-----:                                                       | :-----:                              |
| `true_positive`                    | tp                                                            |                                      |
| `true_negative`                    | tn                                                            |                                      |
| `false_positive`                   | fp                                                            |                                      |
| `false_negative`                   | fn                                                            |                                      |
| `true_positive_rate`               | tp/p                                                          | `sensitivity`,  `recall`, `hit_rate` |
| `true_negative_rate`               | tn/n                                                          | `specificity`,  `selectivity`        |
| `false_positive_rate`              | fp/n                                                          | `fall_out`, `type_I_error`           |
| `false_negative_rate`              | fn/p                                                          | `miss_rate`, `type_II_error`         |
| `precision`                        | tp/(tp + fp)                                                  | `positive_predictive_value`          |
| `negative_predictive_value`        | tn/(tn + fn)                                                  |                                      |
| `false_discovery_rate`             | fp/(fp + tp)                                                  |                                      |
| `false_omission_rate`              | fn/(fn + tn)                                                  |                                      |
| `threat_score`                     | tp/(tp + fn + fp)                                             | `critical_success_index`             |
| `accuracy`                         | (tp + tn)/(p + n)                                             |                                      |
| `balanced_accuracy`                | (tpr + tnr)/2                                                 |                                      |
| `f1_score`                         | 2⋅precision⋅recall/(precision + recall)                       |                                      |
| `fβ_score`                         | (1 + β^{2})⋅precision⋅recall/(β^{2}⋅precision + recall)       |                                      |
| `matthews_correlation_coefficient` | (tp⋅tn + fp⋅fn)/sqrt((tp + fp)⋅(tp + fn)⋅(tn + fp)⋅(tn + fn)) | `mcc`                                |
| `quant`                            | (fn + tn)/(p + n)                                             |                                      |
| `positive_likelihood_ratio`        | tpr/fpr                                                       |                                      |
| `negative_likelihood_ratio`        | fnr/tnr                                                       |                                      |
| `diagnostic_odds_ratio`            | positive_likelihood_ratio/negative_likelihood_ratio           |                                      |

#### Defining own classification metric using `@usermetric`

### Threshold functions

### Curve function

## Todos
 - Write MORE Tests

## License
...

## Credits
...