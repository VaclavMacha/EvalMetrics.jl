function test_curves(; atol = 1e-6)
    n  = 1000
    x  = range(0, 1, length = n)

    f(x) = x <= 0.5 ? 2x : 2x - 1

    y1 = x
    y2 = x./2
    y3 = f.(x)

    @test auc(x, y1) ≈ 0.5  atol = atol
    @test auc(x, y2) ≈ 0.25 atol = atol
    @test auc(x, y3) ≈ 0.5  atol = atol
end