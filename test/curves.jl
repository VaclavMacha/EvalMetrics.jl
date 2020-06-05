import EvalMetrics.Encodings: positive_label, negative_label
import EvalMetrics: apply, curve_points, thresholds


targets = [
    collect(1:10 .>= 6),
    collect(1:10 .>= 8),
    collect(1:10 .>= 3)
]

scores = [
    collect(range(0, 1.0; length=10)),
    collect(range(1.0, 0; length=10)),
    ones(10),
    zeros(10),
    [0.98, 0.26, 0.39, 0.13, 0.3, 0.84, 0.78, 0.48, 0.13, 0.31],
    [0.74, 0.48, 0.23, 0.91, 0.33, 0.92, 0.83, 0.61, 0.68, 0.09]
]

auroc_oracle = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                0.62, 0.35714285714285715, 0.375, 0.6, 0.2857142857142857, 0.5]

auprc_oracle = [1.0, 1.0, 1.0, 0.30436507936507934, 0.16574074074074074,
                0.5927579365079365, 0.75, 0.65, 0.9, 0.75, 0.65, 0.9,
                0.5349999999999999, 0.22222222222222224, 0.6910714285714286,
                0.6477777777777778, 0.20925925925925926, 0.8595734126984127]


set_encoding(OneZero())

encs = [
    OneZero(),
    OneMinusOne(),
    OneTwo(),
    OneVsOne(3,4),
    OneVsOne(:three,:four),
    OneVsOne("three","four"),
    OneVsRest(1, [2,3,4]),
    OneVsRest(:one, [:two, :three]),
    OneVsRest("one", ["two", "three"]),
    RestVsOne([1,2,3], 4),
    RestVsOne([:one, :two], :three),
    RestVsOne(["one", "two"], "three")
]

@testset "curves for $enc encoding" for enc in encs
    global targets

    targets = [recode.(current_encoding(), enc, y) for y in targets]
    set_encoding(enc)

    @testset "check target" begin
        all_neg = fill(negative_label(enc), 10)
        all_pos = fill(positive_label(enc), 10)

        @test_throws ArgumentError auroc(all_neg, rand(10))
        @test_throws ArgumentError auroc(all_pos, rand(10))
        @test_throws ArgumentError auprc(all_neg, rand(10))
        @test_throws ArgumentError auroc(enc, all_neg, rand(10))
        @test_throws ArgumentError auroc(enc, all_pos, rand(10))
        @test_throws ArgumentError auprc(enc, all_neg, rand(10))
    end

    iters = zip(auroc_oracle, auprc_oracle, Iterators.product(targets, scores))
 
    for (auroc_o, auprc_o, (y, s)) in iters
        thres = thresholds(s)

        fpr = false_positive_rate(y, s, thres)
        tpr = true_positive_rate(y, s, thres)
        rec = recall(y, s, thres)
        prec = precision(y, s, thres)

        @testset "curve_points" begin
            @test curve_points(ROCCurve, enc, y, s) == ConfusionMatrix(y, s, thres)
            @test curve_points(PRCurve, enc, y, s) == ConfusionMatrix(y, s, thres)
        end

        @testset "apply" begin
            @test apply(ROCCurve, ConfusionMatrix(y, s, thres)) == (fpr, tpr)
            @test apply(PRCurve, ConfusionMatrix(y, s, thres)) == (rec, prec)
        end

        @testset "auc for ROCCurve" begin
            @test auc(ROCCurve, y, s) ≈ auroc_o
            @test auc(ROCCurve, enc, y, s)≈ auroc_o
            @test auc(ROCCurve, y, s, thres) ≈ auroc_o
            @test auc(ROCCurve, enc, y, s, thres) ≈ auroc_o
        end

        @testset "auroc" begin
            @test auroc(y, s) ≈ auroc_o
            @test auroc(enc, y, s) ≈ auroc_o
            @test auroc(y, s, thres) ≈ auroc_o
            @test auroc(enc, y, s, thres) ≈ auroc_o
        end

        @testset "auc for PRCurve" begin
            @test auc(PRCurve, y, s) ≈ auprc_o
            @test auc(PRCurve, enc, y, s) ≈ auprc_o
            @test auc(PRCurve, y, s, thres) ≈ auprc_o
            @test auc(PRCurve, enc,y, s, thres) ≈ auprc_o
        end

        @testset "auprc" begin
            @test auroc(y, s) ≈ auroc_o
            @test auroc(enc, y, s) ≈ auroc_o
            @test auroc(y, s, thres) ≈ auroc_o
            @test auroc(enc, y, s, thres) ≈ auroc_o
        end
    end
end
