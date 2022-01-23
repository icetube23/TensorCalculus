using TensorCalculus
using Test

@testset "Properties" begin
    t1 = Tensor(rand(3, 3))
    @test size(t1) == (3, 3)
    @test ndims(t1) == 2
    @test length(t1) == 9
    @test eltype(t1) == Float64

    t2 = Tensor(rand(Int32, 3, 3, 3))
    @test size(t2) == (3, 3, 3)
    @test ndims(t2) == 3
    @test length(t2) == 27
    @test eltype(t2) == Int32

    t3 = Tensor(3f0)
    @test size(t3) == ()
    @test ndims(t3) == 0
    @test length(t3) == 1
    @test eltype(t3) == Float32
end

@testset "Equality" begin
    x = rand(2, 3, 3)
    y = x
    t1 = Tensor(x)
    t2 = Tensor(y)
    @test t1 == t2
    @test t1 === t2

    y = copy(x)
    t1 = Tensor(x)
    t2 = Tensor(y)
    @test t1 == t2
    @test t1 !== t2

    y[1, 2, 3] = 4
    t1 = Tensor(x)
    t2 = Tensor(y)
    @test t1 != t2
    @test t1 !== t2
end

@testset "Indexing" begin
    t = Tensor(rand(1, 2, 3, 3))
    @test axes(t) == (Base.OneTo(1), Base.OneTo(2), Base.OneTo(3), Base.OneTo(3))
    @test eachindex(t) == Base.OneTo(18)

    t1 = t[1, :, :, :]
    @test t1 isa Tensor{Float64, 3}
    @test size(t1) == (2, 3, 3)
    @test t1 == t[1, :, :, :]

    t2 = t[:, 2, :, 3]
    @test t2 isa Tensor{Float64, 2}
    @test size(t2) == (1, 3)
    @test t2 == t[:, 2, :, 3]

    t3 = t[1, 2, 3, 2]
    @test t3 isa Tensor{Float64, 0}
    @test size(t3) == ()
    @test t3 == t[1, 2, 3, 2]
    @test_throws BoundsError t3[1]

    t[1, 1, 1, :] = Tensor([1, 2, 3])
    @test t[1, 1, 1, :] == Tensor([1., 2., 3.])

    t[1, 1, 1, 2] = 4
    @test t[1, 1, 1, :] == Tensor([1., 4., 3.])
end