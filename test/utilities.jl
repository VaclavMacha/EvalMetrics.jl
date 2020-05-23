x = sort(rand(123))
ys = [sort(rand(456)), rand()]


@testset "mergesorted" for y in ys
    z, inds = mergesorted(x, y)

    @test z == sort(vcat(x, y))
    @test z[inds] == y
    @test z[setdiff(1:length(z), inds)] == x 
end