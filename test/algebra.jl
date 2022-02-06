@testset "Outer product" begin
    t1 = Tensor([1, 2, 4])
    t2 = Tensor([1.0, 0.5, 0.25])

    # the outer product of two 1-dimensional tensors yields a 2-dimensional tensor
    t3 = t1 ⊗ t2
    @test eltype(t3) === Float64
    @test size(t3) == (size(t1)..., size(t2)...)
    @test t3 == Tensor([
        1.0 0.5 0.25
        2.0 1.0 0.5
        4.0 2.0 1.0
    ])

    t4 = Tensor(rand(3, 3))
    @test t4 ⊗ Tensor(3.14) == t4 * 3.14
    @test t4 ⊗ Tensor([1, 1]) == Tensor(cat(t4.data, t4.data; dims=3))
    @test Tensor(rand(4, 5, 6)) ⊗ Tensor(zeros(3, 2)) == Tensor(zeros(4, 5, 6, 3, 2))

    # you can multiply multiple tensors at once
    t5 = ⊗(Tensor(rand(3, 6)), Tensor(rand(2, 8, 2)), Tensor(rand(9, 1, 7)))
    @test size(t5) == (3, 6, 2, 8, 2, 9, 1, 7)
end

@testset "Inner product" begin
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([5, 2, -3])

    # for 1-dimensional tensors the inner product is equivalent to the vector dot product
    @test t1 ⋅ t2 == Tensor(0)

    t3 = Tensor([1 -1 1;
                 -1 1 -1;
                 1 -1 1])
    @test t1 ⋅ t3 == Tensor([2, -2, 2])
    @test t2 ⋅ t3 == Tensor([0, 0, 0])

    t4 = Tensor(rand(3, 6))
    t5 = Tensor(rand(6, 5))

    # for 2-dimensional tensors the inner product is equivalent to the matrix product
    @test t4 ⋅ t5 ≈ Tensor(t4.data * t5.data)

    # analogously for a matrix-vector product
    t6 = Tensor(rand(6))
    @test t4 ⋅ t6 ≈ Tensor(t4.data * t6.data)
end

# TODO: Test cross product (a.k.a. vector product)
