"""
    true_positive(x::Counts)
    true_positive(target::IntegerVector, predict::RealVector)
    true_positive(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # true positive samples.
"""
true_positive(x::Counts) = x.tp
true_positive(target::IntegerVector, predict::RealVector) =
    true_positive(counts(target, predict))
true_positive(target::IntegerVector, scores::RealVector, threshold::Real) =
    true_positive(counts(target, scores, threshold))


"""
    true_negative(x::Counts)
    true_negative(target::IntegerVector, predict::RealVector)
    true_negative(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # true negative samples.
"""
true_negative(x::Counts) = x.tn
true_negative(target::IntegerVector, predict::RealVector) =
    true_negative(counts(target, predict))
true_negative(target::IntegerVector, scores::RealVector, threshold::Real) =
    true_negative(counts(target, scores, threshold))


"""
    false_positive(x::Counts)
    false_positive(target::IntegerVector, predict::RealVector)
    false_positive(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # false positive samples.
"""
false_positive(x::Counts) = x.fp
false_positive(target::IntegerVector, predict::RealVector) =
    false_positive(counts(target, predict))
false_positive(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_positive(counts(target, scores, threshold))


"""
    false_negative(x::Counts)
    false_negative(target::IntegerVector, predict::RealVector)
    false_negative(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # false negative samples.
"""
false_negative(x::Counts) = x.fn
false_negative(target::IntegerVector, predict::RealVector) =
    false_negative(counts(target, predict))
false_negative(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_negative(counts(target, scores, threshold))


"""
    true_positive_rate(x::Counts)
    true_positive_rate(target::IntegerVector, predict::RealVector)
    true_positive_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns true positive rate `tp/p`.

# Aliases
    sensitivity(...)
    recall(...)
    hit_rate(...)
"""
true_positive_rate(x::Counts) = x.tp/x.p
true_positive_rate(target::IntegerVector, predict::RealVector) =
    true_positive_rate(counts(target, predict))
true_positive_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    true_positive_rate(counts(target, scores, threshold))
sensitivity(x...) = true_positive_rate(x...)
recall(x...) = true_positive_rate(x...)
hit_rate(x...) = true_positive_rate(x...)


"""
    true_negative_rate(x::Counts)
    true_negative_rate(target::IntegerVector, predict::RealVector)
    true_negative_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns true negative rate `tn/n`.

# Aliases
    specificity(...)
    selectivity(...)
"""
true_negative_rate(x::Counts) = x.tn/x.n
true_negative_rate(target::IntegerVector, predict::RealVector) =
    true_negative_rate(counts(target, predict))
true_negative_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    true_negative_rate(counts(target, scores, threshold))
specificity(x...) = true_negative_rate(x...)
selectivity(x...) = true_negative_rate(x...)


"""
    false_positive_rate(x::Counts)
    false_positive_rate(target::IntegerVector, predict::RealVector)
    false_positive_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns false positive rate `fp/n`.

# Aliases
    fall_out(...)
"""
false_positive_rate(x::Counts) = x.fp/x.n
false_positive_rate(target::IntegerVector, predict::RealVector) =
    false_positive_rate(counts(target, predict))
false_positive_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_positive_rate(counts(target, scores, threshold))
fall_out(x...) = false_positive_rate(x...)


"""
    false_negative_rate(x::Counts)
    false_negative_rate(target::IntegerVector, predict::RealVector)
    false_negative_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns false negative rate `fn/p`.

# Aliases
    miss_rate(...)
"""
false_negative_rate(x::Counts) = x.fn/x.p
false_negative_rate(target::IntegerVector, predict::RealVector) =
    false_negative_rate(counts(target, predict))
false_negative_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_negative_rate(counts(target, scores, threshold))
miss_rate(x...) = false_negative_rate(x...)


"""
    positive_predictive_value(x::Counts)
    positive_predictive_value(target::IntegerVector, predict::RealVector)
    positive_predictive_value(target::IntegerVector, scores::RealVector, threshold::Real)

Returns positive predictive value `tp/(tp + fp)`.
# Aliases
    precision(...)
"""
positive_predictive_value(x::Counts) = x.tp/(x.tp + x.fp)
positive_predictive_value(target::IntegerVector, predict::RealVector) =
    positive_predictive_value(counts(target, predict))
positive_predictive_value(target::IntegerVector, scores::RealVector, threshold::Real) =
    positive_predictive_value(counts(target, scores, threshold))
precision(x...) = positive_predictive_value(x...)


"""
    negative_predictive_value(x::Counts)
    negative_predictive_value(target::IntegerVector, predict::RealVector)
    negative_predictive_value(target::IntegerVector, scores::RealVector, threshold::Real)

Returns negative predictive value `tn/(tn + fn)`.
"""
negative_predictive_value(x::Counts) = x.tn/(x.tn + x.fn)
negative_predictive_value(target::IntegerVector, predict::RealVector) =
    negative_predictive_value(counts(target, predict))
negative_predictive_value(target::IntegerVector, scores::RealVector, threshold::Real) =
    negative_predictive_value(counts(target, scores, threshold))


"""
    false_discovery_rate(x::Counts)
    false_discovery_rate(target::IntegerVector, predict::RealVector)
    false_discovery_rate(target::IntegerVector, scores::RealVector, threshold::Real)


Returns false discovery rate `fp/(fp + tp)`.
"""
false_discovery_rate(x::Counts) = x.fp/(x.fp + x.tp)
false_discovery_rate(target::IntegerVector, predict::RealVector) =
    false_discovery_rate(counts(target, predict))
false_discovery_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_discovery_rate(counts(target, scores, threshold))


"""
    false_omission_rate(x::Counts)
    false_omission_rate(target::IntegerVector, predict::RealVector)
    false_omission_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns false omission rate `fn/(fn + tn)`.
"""
false_omission_rate(x::Counts) = x.fn/(x.fn + x.tn)
false_omission_rate(target::IntegerVector, predict::RealVector) =
    false_omission_rate(counts(target, predict))
false_omission_rate(target::IntegerVector, scores::RealVector, threshold::Real) =
    false_omission_rate(counts(target, scores, threshold))


"""
    threat_score(x::Counts)
    threat_score(target::IntegerVector, predict::RealVector)
    threat_score(target::IntegerVector, scores::RealVector, threshold::Real)

Returns threat score `tp/(tp + fn + fp)`.

# Aliases
    critical_success_index(...)
"""
threat_score(x::Counts) = x.tp/(x.tp + x.fn + x.fp)
threat_score(target::IntegerVector, predict::RealVector) =
    threat_score(counts(target, predict))
threat_score(target::IntegerVector, scores::RealVector, threshold::Real) =
    threat_score(counts(target, scores, threshold))
critical_success_index(x...) = threat_score(x...)


"""
    accuracy(x::Counts)
    accuracy(target::IntegerVector, predict::RealVector)
    accuracy(target::IntegerVector, scores::RealVector, threshold::Real)

Returns accuracy `(tp + tn)/(p + n).
"""
accuracy(x::Counts) = (x.tp + x.tn)/(x.p + x.n)
accuracy(target::IntegerVector, predict::RealVector) =
    accuracy(counts(target, predict))
accuracy(target::IntegerVector, scores::RealVector, threshold::Real) =
    accuracy(counts(target, scores, threshold))


"""
    balanced_accuracy(x::Counts)
    balanced_accuracy(target::IntegerVector, predict::RealVector)
    balanced_accuracy(target::IntegerVector, scores::RealVector, threshold::Real)

Returns balanced accuracy `(tpr + fpr)/2`.
"""
balanced_accuracy(x::Counts) = (true_positive_rate(x) + true_negative_rate(x))/2
balanced_accuracy(target::IntegerVector, predict::RealVector) =
    balanced_accuracy(counts(target, predict))
balanced_accuracy(target::IntegerVector, scores::RealVector, threshold::Real) =
    balanced_accuracy(counts(target, scores, threshold))


"""
    f1_score(x::Counts)
    f1_score(target::IntegerVector, predict::RealVector)
    f1_score(target::IntegerVector, scores::RealVector, threshold::Real)

Returns f1 score `2*precision*recall/(precision + recall)`.
"""
f1_score(x::Counts) = 2*precision(x)*recall(x)/(precision(x) + recall(x))
f1_score(target::IntegerVector, predict::RealVector) =
    f1_score(counts(target, predict))
f1_score(target::IntegerVector, scores::RealVector, threshold::Real) =
    f1_score(counts(target, scores, threshold))


"""
    fβ_score(x::Counts; [β::Real = 1])
    fβ_score(target::IntegerVector, predict::RealVector; [β::Real = 1])
    fβ_score(target::IntegerVector, scores::RealVector, threshold::Real; [β::Real = 1])

Returns fβ score `(1 + β^2)*precision*recall/(β^2*precision + recall)`.
"""
fβ_score(x::Counts; β::Real = 1) = (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))
fβ_score(target::IntegerVector, predict::RealVector; β::Real = 1) =
    fβ_score(counts(target, predict); β = β)
fβ_score(target::IntegerVector, scores::RealVector, threshold::Real; β::Real = 1) =
    fβ_score(counts(target, scores, threshold); β = β)


"""
    matthews_correlation_coefficient(x::Counts)
    matthews_correlation_coefficient(target::IntegerVector, predict::RealVector)
    matthews_correlation_coefficient(target::IntegerVector, scores::RealVector, threshold::Real)

Returns Matthews correlation coefficient `(tp*tn - fp*fn)/sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))`.

# Aliases
    mcc(...)
"""
matthews_correlation_coefficient(x::Counts) =
    (x.tp*x.tn + x.fp*x.fn)/sqrt((x.tp + x.fp)*(x.tp + x.fn)*(x.tn + x.fp)*(x.tn + x.fn))
matthews_correlation_coefficient(target::IntegerVector, predict::RealVector) =
    matthews_correlation_coefficient(counts(target, predict))
matthews_correlation_coefficient(target::IntegerVector, scores::RealVector, threshold::Real) =
    matthews_correlation_coefficient(counts(target, scores, threshold))
mcc(x...) = matthews_correlation_coefficient(x...)


"""
    quantile(x::Counts)
    quantile(target::IntegerVector, predict::RealVector)
    quantile(target::IntegerVector, scores::RealVector, threshold::Real)

Returns quantile `(x.fn + x.tn)/(x.p + x.n)`.
"""
quantile(x::Counts) = (x.fn + x.tn)/(x.p + x.n)
quantile(target::IntegerVector, predict::RealVector) =
    quantile(counts(target, predict))
quantile(target::IntegerVector, scores::RealVector, threshold::Real) =
    quantile(counts(target, scores, threshold))


"""
    positive_likelihood_ratio(x::Counts)
    positive_likelihood_ratio(target::IntegerVector, predict::RealVector)
    positive_likelihood_ratio(target::IntegerVector, scores::RealVector, threshold::Real)

Returns positive likelyhood ratio `tpr/fpr`.
"""
positive_likelihood_ratio(x::Counts) = true_positive_rate(x)/false_positive_rate(x)
positive_likelihood_ratio(target::IntegerVector, predict::RealVector) =
    positive_likelihood_ratio(counts(target, predict))
positive_likelihood_ratio(target::IntegerVector, scores::RealVector, threshold::Real) =
    positive_likelihood_ratio(counts(target, scores, threshold))


"""
    negative_likelihood_ratio(x::Counts)
    negative_likelihood_ratio(target::IntegerVector, predict::RealVector)
    negative_likelihood_ratio(target::IntegerVector, scores::RealVector, threshold::Real)

Returns negative likelyhood ratio `fnr/tnr`.
"""
negative_likelihood_ratio(x::Counts) = false_negative_rate(x)/true_negative_rate(x)
negative_likelihood_ratio(target::IntegerVector, predict::RealVector) =
    negative_likelihood_ratio(counts(target, predict))
negative_likelihood_ratio(target::IntegerVector, scores::RealVector, threshold::Real) =
    negative_likelihood_ratio(counts(target, scores, threshold))


"""
    diagnostic_odds_ratio(x::Counts)
    diagnostic_odds_ratio(target::IntegerVector, predict::RealVector)
    diagnostic_odds_ratio(target::IntegerVector, scores::RealVector, threshold::Real)

Returns diagnostic odds ratio `tpr*tnr/(fpr*fnr)`.
"""
diagnostic_odds_ratio(x::Counts) = positive_likelihood_ratio(x)/negative_likelihood_ratio(x)
diagnostic_odds_ratio(target::IntegerVector, predict::RealVector) =
    diagnostic_odds_ratio(counts(target, predict))
diagnostic_odds_ratio(target::IntegerVector, scores::RealVector, threshold::Real) =
    diagnostic_odds_ratio(counts(target, scores, threshold))