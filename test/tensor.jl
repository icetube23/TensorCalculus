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

@testset "Printing" begin
    io = IOBuffer()

    t1 = Tensor(2)
    show(io, t1)
    @test String(take!(io)) == "Tensor{Int64, 0}(2)"
    show(io, "text/plain", t1)
    @test String(take!(io)) == """scalar Tensor{Int64, 0}:\n\
                                  2"""

    t2 = Tensor(Vector{Float64}(1:4))
    show(io, t2)
    @test String(take!(io)) == "Tensor{Float64, 1}([1.0, 2.0, 3.0, 4.0])"
    show(io, "text/plain", t2)
    @test String(take!(io)) == """4-element Tensor{Float64, 1}:\n \
                                   1.0\n \
                                   2.0\n \
                                   3.0\n \
                                   4.0"""
    
    t3 = Tensor(reshape(Vector{Int32}(1:9), 3, 3))
    show(io, t3)
    @test String(take!(io)) == "Tensor{Int32, 2}(Int32[1 4 7; 2 5 8; 3 6 9])"
    show(io, "text/plain", t3)
    @test String(take!(io)) == """3×3 Tensor{Int32, 2}:\n \
                                   1  4  7\n \
                                   2  5  8\n \
                                   3  6  9"""
    
    t4 = Tensor(reshape(Vector{Float32}(1:8), 2, 2, 2))
    show(io, t4)
    @test String(take!(io)) == "Tensor{Float32, 3}([1.0 3.0; 2.0 4.0;;; 5.0 7.0; 6.0 8.0])"
    show(io, "text/plain", t4)
    @test String(take!(io)) == """2×2×2 Tensor{Float32, 3}:\n\
                                  [:, :, 1] =\n \
                                   1.0  3.0\n \
                                   2.0  4.0\n\
                                  \n\
                                  [:, :, 2] =\n \
                                   5.0  7.0\n \
                                   6.0  8.0"""
end