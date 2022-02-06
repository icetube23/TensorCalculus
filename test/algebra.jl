@testset "Outer product" begin
    t1 = Tensor([1, 2, 4])
    t2 = Tensor([1.0, 0.5, 0.25])

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

    # TODO: add some more complex test cases for better coverage
end

# TODO: Tensor scalar product (a.k.a. inner product)

# TODO: Test cross product (a.k.a. vector product)
