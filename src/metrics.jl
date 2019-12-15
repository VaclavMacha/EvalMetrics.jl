"""
    true_positive(x::Counts)
    true_positive(target::IntegerVector, predict::RealVector)
    true_positive(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # true positive samples.
"""
true_positive(x::Counts) = x.tp
true_positive(tr::IntegerVector, pr::RealVector) =
    true_positive(counts(tr, pr))
true_positive(tr::IntegerVector, sc::RealVector, t::Real) =
    true_positive(counts(tr, sc, t))


"""
    true_negative(x::Counts)
    true_negative(target::IntegerVector, predict::RealVector)
    true_negative(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # true negative samples.
"""
true_negative(x::Counts) = x.tn
true_negative(tr::IntegerVector, pr::RealVector) =
    true_negative(counts(tr, pr))
true_negative(tr::IntegerVector, sc::RealVector, t::Real) =
    true_negative(counts(tr, sc, t))


"""
    false_positive(x::Counts)
    false_positive(target::IntegerVector, predict::RealVector)
    false_positive(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # false positive samples.
"""
false_positive(x::Counts) = x.fp
false_positive(tr::IntegerVector, pr::RealVector) =
    false_positive(counts(tr, pr))
false_positive(tr::IntegerVector, sc::RealVector, t::Real) =
    false_positive(counts(tr, sc, t))


"""
    false_negative(x::Counts)
    false_negative(target::IntegerVector, predict::RealVector)
    false_negative(target::IntegerVector, scores::RealVector, threshold::Real)

Returns # false negative samples.
"""
false_negative(x::Counts) = x.fn
false_negative(tr::IntegerVector, pr::RealVector) =
    false_negative(counts(tr, pr))
false_negative(tr::IntegerVector, sc::RealVector, t::Real) =
    false_negative(counts(tr, sc, t))


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
true_positive_rate(tr::IntegerVector, pr::RealVector) =
    true_positive_rate(counts(tr, pr))
true_positive_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    true_positive_rate(counts(tr, sc, t))
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
true_negative_rate(tr::IntegerVector, pr::RealVector) =
    true_negative_rate(counts(tr, pr))
true_negative_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    true_negative_rate(counts(tr, sc, t))
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
false_positive_rate(tr::IntegerVector, pr::RealVector) =
    false_positive_rate(counts(tr, pr))
false_positive_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    false_positive_rate(counts(tr, sc, t))
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
false_negative_rate(tr::IntegerVector, pr::RealVector) =
    false_negative_rate(counts(tr, pr))
false_negative_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    false_negative_rate(counts(tr, sc, t))
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
positive_predictive_value(tr::IntegerVector, pr::RealVector) =
    positive_predictive_value(counts(tr, pr))
positive_predictive_value(tr::IntegerVector, sc::RealVector, t::Real) =
    positive_predictive_value(counts(tr, sc, t))
precision(x...) = positive_predictive_value(x...)


"""
    negative_predictive_value(x::Counts)
    negative_predictive_value(target::IntegerVector, predict::RealVector)
    negative_predictive_value(target::IntegerVector, scores::RealVector, threshold::Real)

Returns negative predictive value `tn/(tn + fn)`.
"""
negative_predictive_value(x::Counts) = x.tn/(x.tn + x.fn)
negative_predictive_value(tr::IntegerVector, pr::RealVector) =
    negative_predictive_value(counts(tr, pr))
negative_predictive_value(tr::IntegerVector, sc::RealVector, t::Real) =
    negative_predictive_value(counts(tr, sc, t))


"""
    false_discovery_rate(x::Counts)
    false_discovery_rate(target::IntegerVector, predict::RealVector)
    false_discovery_rate(target::IntegerVector, scores::RealVector, threshold::Real)


Returns false discovery rate `fp/(fp + tp)`.
"""
false_discovery_rate(x::Counts) = x.fp/(x.fp + x.tp)
false_discovery_rate(tr::IntegerVector, pr::RealVector) =
    false_discovery_rate(counts(tr, pr))
false_discovery_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    false_discovery_rate(counts(tr, sc, t))


"""
    false_omission_rate(x::Counts)
    false_omission_rate(target::IntegerVector, predict::RealVector)
    false_omission_rate(target::IntegerVector, scores::RealVector, threshold::Real)

Returns false omission rate `fn/(fn + tn)`.
"""
false_omission_rate(x::Counts) = x.fn/(x.fn + x.tn)
false_omission_rate(tr::IntegerVector, pr::RealVector) =
    false_omission_rate(counts(tr, pr))
false_omission_rate(tr::IntegerVector, sc::RealVector, t::Real) =
    false_omission_rate(counts(tr, sc, t))


"""
    threat_score(x::Counts)
    threat_score(target::IntegerVector, predict::RealVector)
    threat_score(target::IntegerVector, scores::RealVector, threshold::Real)

Returns threat score `tp/(tp + fn + fp)`.

# Aliases
    critical_success_index(...)
"""
threat_score(x::Counts) = x.tp/(x.tp + x.fn + x.fp)
threat_score(tr::IntegerVector, pr::RealVector) =
    threat_score(counts(tr, pr))
threat_score(tr::IntegerVector, sc::RealVector, t::Real) =
    threat_score(counts(tr, sc, t))
critical_success_index(x...) = threat_score(x...)


"""
    accuracy(x::Counts)
    accuracy(target::IntegerVector, predict::RealVector)
    accuracy(target::IntegerVector, scores::RealVector, threshold::Real)

Returns accuracy `(tp + tn)/(p + n).
"""
accuracy(x::Counts) = (x.tp + x.tn)/(x.p + x.n)
accuracy(tr::IntegerVector, pr::RealVector) =
    accuracy(counts(tr, pr))
accuracy(tr::IntegerVector, sc::RealVector, t::Real) =
    accuracy(counts(tr, sc, t))


"""
    balanced_accuracy(x::Counts)
    balanced_accuracy(target::IntegerVector, predict::RealVector)
    balanced_accuracy(target::IntegerVector, scores::RealVector, threshold::Real)

Returns balanced accuracy `(tpr + fpr)/2`.
"""
balanced_accuracy(x::Counts) = (true_positive_rate(x) + true_negative_rate(x))/2
balanced_accuracy(tr::IntegerVector, pr::RealVector) =
    balanced_accuracy(counts(tr, pr))
balanced_accuracy(tr::IntegerVector, sc::RealVector, t::Real) =
    balanced_accuracy(counts(tr, sc, t))


"""
    f1_score(x::Counts)
    f1_score(target::IntegerVector, predict::RealVector)
    f1_score(target::IntegerVector, scores::RealVector, threshold::Real)

Returns f1 score `2*precision*recall/(precision + recall)`.
"""
f1_score(x::Counts) = 2*precision(x)*recall(x)/(precision(x) + recall(x))
f1_score(tr::IntegerVector, pr::RealVector) =
    f1_score(counts(tr, pr))
f1_score(tr::IntegerVector, sc::RealVector, t::Real) =
    f1_score(counts(tr, sc, t))


"""
    fβ_score(x::Counts; [β::Real = 1])
    fβ_score(target::IntegerVector, predict::RealVector; [β::Real = 1])
    fβ_score(target::IntegerVector, scores::RealVector, threshold::Real; [β::Real = 1])

Returns fβ score `(1 + β^2)*precision*recall/(β^2*precision + recall)`.
"""
fβ_score(x::Counts; β::Real = 1) = (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))
fβ_score(tr::IntegerVector, pr::RealVector; β::Real = 1) =
    fβ_score(counts(tr, pr); β = β)
fβ_score(tr::IntegerVector, sc::RealVector, t::Real; β::Real = 1) =
    fβ_score(counts(tr, sc, t); β = β)


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
matthews_correlation_coefficient(tr::IntegerVector, pr::RealVector) =
    matthews_correlation_coefficient(counts(tr, pr))
matthews_correlation_coefficient(tr::IntegerVector, sc::RealVector, t::Real) =
    matthews_correlation_coefficient(counts(tr, sc, t))
mcc(x...) = matthews_correlation_coefficient(x...)


"""
    quantile(x::Counts)
    quantile(target::IntegerVector, predict::RealVector)
    quantile(target::IntegerVector, scores::RealVector, threshold::Real)

Returns quantile `(x.fn + x.tn)/(x.p + x.n)`.
"""
quantile(x::Counts) = (x.fn + x.tn)/(x.p + x.n)
quantile(tr::IntegerVector, pr::RealVector) =
    quantile(counts(tr, pr))
quantile(tr::IntegerVector, sc::RealVector, t::Real) =
    quantile(counts(tr, sc, t))
