"""
    $(SIGNATURES) 

Returns # true positive samples.
"""
@usermetric true_positive(x::Counts) = x.tp


"""
    $(SIGNATURES) 

Returns # true negative samples.
"""
@usermetric true_negative(x::Counts) = x.tn


"""
    $(SIGNATURES) 

Returns # false positive samples.
"""
@usermetric false_positive(x::Counts) = x.fp


"""
    $(SIGNATURES) 

Returns # false negative samples.
"""
@usermetric false_negative(x::Counts) = x.fn


"""
    $(SIGNATURES) 

Returns true positive rate `tp/p`.
Aliases: `sensitivity`,  `recall`, `hit_rate`.
"""
@usermetric true_positive_rate(x::Counts) = x.tp/x.p
const sensitivity = true_positive_rate
const recall      = true_positive_rate
const hit_rate    = true_positive_rate


"""
    $(SIGNATURES) 

Returns true negative rate `tn/n`.
Aliases: `specificity`,  `selectivity`.
"""
@usermetric true_negative_rate(x::Counts) = x.tn/x.n
const specificity = true_negative_rate
const selectivity = true_negative_rate


"""
    $(SIGNATURES) 

Returns false positive rate `fp/n`.
Aliases: `fall_out`, `type_I_error`.
"""
@usermetric false_positive_rate(x::Counts) = x.fp/x.n
const fall_out     = false_positive_rate
const type_I_error = false_positive_rate


"""
    $(SIGNATURES) 

Returns false negative rate `fn/p`.
Aliases: `miss_rate`, `type_II_error`.
"""
@usermetric false_negative_rate(x::Counts) = x.fn/x.p
const miss_rate     = false_negative_rate
const type_II_error = false_negative_rate


"""
    $(SIGNATURES) 

Returns precision `tp/(tp + fp)`.
Aliases: `positive_predictive_value`.
"""
@usermetric precision(x::Counts) = x.tp/(x.tp + x.fp)
const positive_predictive_value = precision


"""
    $(SIGNATURES) 

Returns negative predictive value `tn/(tn + fn)`.
"""
@usermetric negative_predictive_value(x::Counts) = x.tn/(x.tn + x.fn)


"""
    $(SIGNATURES) 

Returns false discovery rate `fp/(fp + tp)`.
"""
@usermetric false_discovery_rate(x::Counts) = x.fp/(x.fp + x.tp)


"""
    $(SIGNATURES) 

Returns false omission rate `fn/(fn + tn)`.
"""
@usermetric false_omission_rate(x::Counts) = x.fn/(x.fn + x.tn)


"""
    $(SIGNATURES) 

Returns threat score `tp/(tp + fn + fp)`.
Aliases: `critical_success_index`.
"""
@usermetric threat_score(x::Counts) = x.tp/(x.tp + x.fn + x.fp)
const critical_success_index = threat_score


"""
    $(SIGNATURES) 

Returns accuracy `(tp + tn)/(p + n).
"""
@usermetric accuracy(x::Counts) = (x.tp + x.tn)/(x.p + x.n)


"""
    $(SIGNATURES) 

Returns balanced accuracy `(tpr + fpr)/2`.
"""
@usermetric balanced_accuracy(x::Counts) = (true_positive_rate(x) + true_negative_rate(x))/2


"""
    $(SIGNATURES) 

Returns f1 score `2*precision*recall/(precision + recall)`.
"""
@usermetric f1_score(x::Counts) = 2*precision(x)*recall(x)/(precision(x) + recall(x))


"""
    $(SIGNATURES) 

Returns fβ score `(1 + β^2)*precision*recall/(β^2*precision + recall)`.
"""
@usermetric function fβ_score(x::Counts; β::Real = 1) 
    (1 + β^2)*precision(x)*recall(x)/(β^2*precision(x) + recall(x))
end


"""
    $(SIGNATURES) 

Returns Matthews correlation coefficient `(tp*tn - fp*fn)/sqrt((tp+fp)(tp+fn)(tn+fp)(tn+fn))`.
Aliases: ` mcc`.
"""
@usermetric function matthews_correlation_coefficient(x::Counts)
    (x.tp*x.tn + x.fp*x.fn)/sqrt((x.tp + x.fp)*(x.tp + x.fn)*(x.tn + x.fp)*(x.tn + x.fn))
end
const mcc = matthews_correlation_coefficient


"""
    $(SIGNATURES) 

Returns quant `(x.fn + x.tn)/(x.p + x.n)`.
"""
@usermetric quant(x::Counts) = (x.fn + x.tn)/(x.p + x.n)


"""
    $(SIGNATURES) 

Returns topquant `1 - quant`.
"""
@usermetric topquant(x::Counts) = 1 - quant(x)


"""
    $(SIGNATURES) 

Returns positive likelyhood ratio `tpr/fpr`.
"""
@usermetric positive_likelihood_ratio(x::Counts) = true_positive_rate(x)/false_positive_rate(x)


"""
    $(SIGNATURES) 

Returns negative likelyhood ratio `fnr/tnr`.
"""
@usermetric negative_likelihood_ratio(x::Counts) = false_negative_rate(x)/true_negative_rate(x)


"""
    $(SIGNATURES) 

Returns diagnostic odds ratio `tpr*tnr/(fpr*fnr)`.
"""
@usermetric diagnostic_odds_ratio(x::Counts) = positive_likelihood_ratio(x)/negative_likelihood_ratio(x)