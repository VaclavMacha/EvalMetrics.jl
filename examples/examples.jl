# Example 1
using EvalMetrics

target  = [  1,   1,   1,   1,   0,   0,   0,   0,   0,   0];
predict = [  1,   1,   0,   0,   1,   1,   0,   0,   0,   0];
scores  = [0.7, 0.8, 0.3, 0.2, 0.8, 0.9, 0.2, 0.1, 0.2, 0.3];
thres   = 0.4;

counts(target, predict)
counts(target, scores, thres)
counts(target, scores, [thres, thres])


# Example 2
c = counts(target, predict);
precision(c)
precision(target, predict)
precision(target, scores, thres)
precision([c, c])
precision(target, scores, [thres, thres])


# Example 3
import EvalMetrics: @usermetric

@usermetric my_precision(x::Counts) = x.tp/(x.tp + x.fp)

my_precision(c)
my_precision(target, predict)
my_precision(target, scores, thres)
my_precision([c, c])
my_precision(target, scores, [thres, thres])


# Example 4
f(x::Counts) = 1
@usermetric f


# Example 5
@usermetric my_metric(x::Counts, y::Array) = y
my_metric([c], [1,2,3])
my_metric([c, c], [1,2,3])
my_metric([c, c], [[1,2,3]])


# Example 5
@usermetric my_metric_kwargs(x::Counts; y::Array = []) = y
my_metric_kwargs([c];    y = [1,2,3])
my_metric_kwargs([c, c], y = [1,2,3])
my_metric_kwargs([c, c], y = [[1,2,3]])


# Example 6
target = 1:6 .>= 3;
scores = 1:6;
t1 = thresholds(scores);
t2 = thresholds(scores, 8; reduced = true,  zerorecall = true);
t3 = thresholds(scores, 8; reduced = true,  zerorecall = false);
t4 = thresholds(scores, 8; reduced = false, zerorecall = true);
t5 = thresholds(scores, 8; reduced = false, zerorecall = false);

hcat(t1, t2, t3,
     recall(target, scores, t1),
     recall(target, scores, t2),
     recall(target, scores, t3))
hcat(t4, t5,
     recall(target, scores, t4),
     recall(target, scores, t5))


# Examples 7
using Test, Random
Random.seed!(1234);

target = rand(0:1, 10000);
scores = rand(10000);
rate   = 0.2

t1 = threshold_at_tpr(target, scores, rate);
tpr = true_positive_rate(target, scores, t1)

t2  = threshold_at_tnr(target, scores, rate);
tnr = true_negative_rate(target, scores, t2)

t3  = threshold_at_fpr(target, scores, rate);
fpr = false_positive_rate(target, scores, t3)

t4  = threshold_at_fnr(target, scores, rate);
fnr = false_negative_rate(target, scores, t4)

@testset "test thresholds_at_" begin
    @test tpr_est >= rate > true_positive_rate(target, scores, t1  + eps())
    @test tnr_est >= rate > true_negative_rate(target, scores, t2  - eps())
    @test fpr_est <= rate < false_positive_rate(target, scores, t3 - eps())
    @test fnr_est <= rate < false_negative_rate(target, scores, t4 + eps())
end;

# Example 8
x = 0:0.1:1
y = 0:0.1:1
auc(x,y)





