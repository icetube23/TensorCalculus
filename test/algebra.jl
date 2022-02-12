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
    # for 1-dimensional tensors the inner product is equivalent to the vector dot product
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([5, 2, -3])
    @test t1 ⋅ t2 == Tensor(0)

    t3 = Tensor([
        1 -1 1
        -1 1 -1
        1 -1 1
    ])
    @test t1 ⋅ t3 == Tensor([2, -2, 2])
    @test t2 ⋅ t3 == Tensor([0, 0, 0])

    # for 2-dimensional tensors the inner product is equivalent to the matrix product
    t4 = Tensor(rand(3, 6))
    t5 = Tensor(rand(6, 5))
    @test t4 ⋅ t5 ≈ Tensor(t4.data * t5.data)

    # analogously for a matrix-vector product
    t6 = Tensor(rand(6))
    @test t4 ⋅ t6 ≈ Tensor(t4.data * t6.data)
end

@testset "Contraction" begin
    a1 = reshape(Vector(1:12), 2, 2, 3)
    t1 = Tensor(a1)
    @test contract(t1, 1, 2) == Tensor([5, 13, 21])
    @test contract(t1, 1, 2) == contract(t1, 2, 1)

    a2 = rand(6, 3, 6, 2)
    t2 = Tensor(a2)
    t3 = contract(t2, 1, 3)
    @test size(t3) == (3, 2)
    @test t3 == Tensor(
        [
            sum(a2[i, 1, i, 1] for i in axes(a2, 1)) sum(a2[i, 1, i, 2] for i in axes(a2, 1))
            sum(a2[i, 2, i, 1] for i in axes(a2, 1)) sum(a2[i, 2, i, 2] for i in axes(a2, 1))
            sum(a2[i, 3, i, 1] for i in axes(a2, 1)) sum(a2[i, 3, i, 2] for i in axes(a2, 1))
        ],
    )

    # contracting is only possible along distinct, valid dimensions of the same size
    @test_throws ArgumentError contract(t1, 2, 2) # not distinct
    @test_throws BoundsError contract(t1, 2, 4) # 4 not a valid dimension for rank 3 tensor
    @test_throws DimensionMismatch contract(t1, 2, 3) # dimensions 2 and 3 are not equal

    # for rank 2 tensors we can use the trace method instead of contract
    @test trace(t1[:, :, 3]) == Tensor(21)
    @test trace(t1[2, :, 1:2]) == Tensor(10)
    @test_throws DimensionMismatch trace(t1[2, :, :])
    @test trace(t2[:, 2, :, 1]) == contract(t2[:, 2, :, 1], 1, 2)
    @test trace(t2[:, 2, :, 1]) == contract(t2[:, 2, :, 1], 2, 1)
end

@testset "Überschiebung" begin
    t1 = Tensor(reshape(Vector(1:12), 2, 3, 2))
    t2 = Tensor([
        1.1 2.2 3.3
        -3.3 -2.2 -1.1
    ])

    t3 = pushover(t1, t2, 2, 2)
    @test size(t3) == (2, 2, 2)
    @test t3 ≈ Tensor(reshape([24.2, 30.8, 63.8, 70.4, -15.4, -22, -55, -61.6], 2, 2, 2))

    t4 = pushover(t1, t2, 1, 1)
    @test size(t4) == (3, 2, 3)
    @test t4 ≈ Tensor(
        reshape(
            [
                -5.5,
                -9.9,
                -14.3,
                -18.7,
                -23.1,
                -27.5,
                -2.2,
                -2.2,
                -2.2,
                -2.2,
                -2.2,
                -2.2,
                1.1,
                5.5,
                9.9,
                14.3,
                18.7,
                23.1,
            ],
            3,
            2,
            3,
        ),
    )

    t5 = pushover(t1, t2, 3, 1)
    @test size(t5) == (2, 3, 3)
    @test t5 ≈ Tensor(
        reshape(
            [
                -22,
                -24.2,
                -26.4,
                -28.6,
                -30.8,
                -33,
                -13.2,
                -13.2,
                -13.2,
                -13.2,
                -13.2,
                -13.2,
                -4.4,
                -2.2,
                0,
                2.2,
                4.4,
                6.6,
            ],
            2,
            3,
            3,
        ),
    )

    # push-over is only possible along valid dimensions of the same size
    @test_throws BoundsError pushover(t1, t2, 3, 3)
    @test_throws DimensionMismatch pushover(t1, t2, 1, 2)

    # push-over satifies many interesting tensor-algebraic identities
    t6 = Tensor(rand(4, 7, 5))
    t7 = Tensor(rand(6, 3, 4))
    @test pushover(t6, t7, 1, 3) == permutedims(pushover(t7, t6, 3, 1), (3, 4, 1, 2))
    @test pushover(t6, t7, 1, 3) == contract(t6 ⊗ t7, 1, 6)
    @test pushover(t6, t7, 1, 3) == permutedims(t6, (2, 3, 1)) ⋅ permutedims(t7, (3, 1, 2))
end

# TODO: Test cross product (a.k.a. vector product)
