n = 1000
targets = rand(0:1, n)
scores = rand(n)
thres = 0.7
predicts = scores .>= thres

p  = sum(targets .== 1)
n  = sum(targets .== 0)
tp = sum(targets .* predicts)
tn = sum((1 .- targets) .* (1 .- predicts))
fp = sum((1 .- targets) .* predicts)
fn = sum(targets .* (1 .- predicts))

cm = ConfusionMatrix(p, n, tp, tn, fp, fn)
cm2 = ConfusionMatrix(2*p, 2*n, 2*tp, 2*tn, 2*fp, 2*fn)

set_encoding(OneZero())

encs = [
    OneZero(),
    OneMinusOne(),
    OneTwo(),
    OneVsRest(1, [2,3,4]),
    OneVsRest(:one, [:two, :three]),
    OneVsRest("one", ["two", "three"]),
    RestVsOne([1,2,3], 4),
    RestVsOne([:one, :two], :three),
    RestVsOne(["one", "two"], "three")
] 


@testset "ConfusionMatrix +" begin
    @test cm + cm == cm2
end


@testset "ConfusionMatrix constructors for $enc encoding" for enc in encs
    global targets, predicts

    targets = recode.(current_encoding(), enc, targets)
    predicts = recode.(current_encoding(), enc, predicts)
    set_encoding(enc)


    @test ConfusionMatrix(targets, predicts) == cm
    @test ConfusionMatrix(enc, targets, predicts) == cm
    @test ConfusionMatrix(targets, scores, thres) == cm
    @test ConfusionMatrix(enc, targets, scores, thres) == cm
    @test ConfusionMatrix(targets, scores, [thres]) == [cm]
    @test ConfusionMatrix(enc, targets, scores, [thres]) == [cm]
end