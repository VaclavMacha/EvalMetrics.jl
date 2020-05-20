function test_auc_trapezoidal(; atol=1e-6)
    n  = 1000
    x  = range(0, 1, length = n)

    f(x) = x <= 0.5 ? 2x : 2x - 1

    y1 = x
    y2 = x./2
    y3 = f.(x)

    @test auc_trapezoidal(x, y1) ≈ 0.5  atol = atol
    @test auc_trapezoidal(x, y2) ≈ 0.25 atol = atol
    @test auc_trapezoidal(x, y3) ≈ 0.5  atol = atol
end

function test_auc()
    y1 = [0,0,0,0,0,1,1,1,1,1]
    c1 = (0, 1)
    y2 = falses(10); y2[8:10] .= true
    c2 = (false, true)
    y3 = [:a,:d,:b,:c,:b,:c,:b,:c,:b,:b]
    c3 = ([:a, :d], [:b, :c])

    s1 = range(0, 1.0; length=10) |> collect
    s2 = s1 |> reverse
    s3 = ones(10)
    s4 = zeros(10)
    s5 = [0.98, 0.26, 0.39, 0.13, 0.3, 0.84, 0.78, 0.48, 0.13, 0.31]
    s6 = [0.74, 0.48, 0.23, 0.91, 0.33, 0.92, 0.83, 0.61, 0.68, 0.09]

    auroc_oracle = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                    0.62, 0.35714285714285715, 0.375, 0.6, 0.2857142857142857, 0.5]
    auprc_oracle = [1.0, 1.0, 1.0, 0.30436507936507934, 0.16574074074074074,
                    0.5927579365079365, 0.75, 0.65, 0.9, 0.75, 0.65, 0.9,
                    0.5349999999999999, 0.22222222222222224, 0.6910714285714286,
                    0.6477777777777778, 0.20925925925925926, 0.8595734126984127]

    for (auroc_o, auprc_o, ((c,y), s)) in zip(auroc_oracle, auprc_oracle,
                                          Iterators.product([(c1,y1), (c2,y2), (c3,y3)], [s1,s2,s3,s4,s5,s6]))
        @test auc(ROCCurve, y, s; classes=c) == auroc(y, s; classes=c) ≈ auroc_o
        @test auc(PRCurve, y, s; classes=c) == auprc(y, s; classes=c) ≈ auprc_o
    end

    @test_throws ArgumentError auroc(zeros(Int, 10), rand(10))
    @test_throws ArgumentError auroc(ones(Int, 10), rand(10))
    @test_throws ArgumentError auprc(zeros(Int, 10), rand(10))
end
