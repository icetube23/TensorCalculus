@testset "Scalar comparisons" begin
    t = Tensor(3.14)

    # compare against larger scalar
    @test t < 4
    @test t <= 4
    @test !(t > 4)
    @test !(t >= 4)
    @test t != 4

    @test 4 > t
    @test 4 >= t
    @test !(4 < t)
    @test !(4 <= t)
    @test 4 != t

    # compare against smaller scalar
    @test !(t < 3)
    @test !(t <= 3)
    @test t > 3
    @test t >= 3
    @test t != 3

    @test !(3 > t)
    @test !(3 >= t)
    @test 3 < t
    @test 3 <= t
    @test 3 != t

    # compare against equal scalar
    @test !(t < 3.14)
    @test t <= 3.14
    @test !(t > 3.14)
    @test t >= 3.14
    @test t == 3.14

    @test !(3.14 > t)
    @test 3.14 >= t
    @test !(3.14 < t)
    @test 3.14 <= t
    @test 3.14 == t

    # comparisons between scalar tensors also work
    @test t < Tensor(4)
    @test t <= Tensor(4)
    @test !(t < Tensor(3))
    @test !(t <= Tensor(3))
    @test t == Tensor(3.14)

    # non-equality comparisons throw if tensor isn't scalar
    t = Tensor([2.71, 3.14])

    @test_throws MethodError t < 4
    @test_throws MethodError t <= 4
    @test_throws MethodError !(t > 4)
    @test_throws MethodError !(t >= 4)

    @test_throws MethodError 4 > t
    @test_throws MethodError 4 >= t
    @test_throws MethodError !(4 < t)
    @test_throws MethodError !(4 <= t)

    @test t != 4
    @test 4 != t
end

@testset "Scalar operations" begin
    # + - * / ÷ \ ^ %
    @test Tensor(2) + 0.4 == Tensor(2.4)
    @test 3 + Tensor(1.5) == Tensor(4.5)
    @test Tensor(5) + Tensor(4) == Tensor(9)

    @test Tensor(2) - 0.4 == Tensor(1.6)
    @test 3 - Tensor(1.5) == Tensor(1.5)
    @test Tensor(5) - Tensor(4) == Tensor(1)

    @test Tensor(2) * 0.4 == Tensor(0.8)
    @test 3 * Tensor(1.5) == Tensor(4.5)
    @test Tensor(5) * Tensor(4) == Tensor(20)

    @test Tensor(2) / 0.4 == Tensor(5.0)
    @test 3 / Tensor(1.5) == Tensor(2.0)
    @test Tensor(5) / Tensor(4) == Tensor(1.25)

    @test Tensor(2) ÷ 0.4 == Tensor(4.0)
    @test 3 ÷ Tensor(1.5) == Tensor(2.0)
    @test Tensor(5) ÷ Tensor(4) == Tensor(1)

    @test Tensor(2) \ 0.4 == Tensor(0.2)
    @test 3 \ Tensor(1.5) == Tensor(0.5)
    @test Tensor(5) \ Tensor(4) == Tensor(0.8)

    @test Tensor(2)^0.4 ≈ Tensor(1.3195079)
    @test 3^Tensor(1.5) ≈ Tensor(5.1961524)
    @test Tensor(5)^Tensor(4) == Tensor(625)

    # floating point error
    @test Tensor(2) % 0.4 ≈ Tensor(0.399999999)
    @test 3 % Tensor(1.5) == Tensor(0.0)
    @test Tensor(5) % Tensor(4) == Tensor(1)
end

@testset "Non-scalar operations" begin
    t1 = Tensor([
        1 2 3
        4 5 6
    ])
    t2 = Tensor([
        1 0 1
        0 1 0
    ])

    # unary +, - operators
    @test +t1 == Tensor([
        1 2 3
        4 5 6
    ])
    @test -t1 == Tensor([
        -1 -2 -3
        -4 -5 -6
    ])

    # binary +, - operators
    @test t1 + t2 == Tensor([
        2 2 4
        4 6 6
    ])
    @test t1 + t2 == t2 + t1

    @test t1 - t2 == Tensor([
        0 2 2
        4 4 6
    ])
    @test t1 - t2 == -(t2 - t1)

    # size of tensors to be added or subtracted has to match or the methods will throw
    t3 = Tensor([
        1 4
        2 5
        3 6
    ])

    @test_throws DimensionMismatch t1 + t3
    @test_throws DimensionMismatch t3 + t2
    @test_throws DimensionMismatch t1 - t3
    @test_throws DimensionMismatch t3 - t2
end

@testset "Non-scalar-scalar arithmetic" begin
    t1 = Tensor(rand(3, 3))
    t2 = Tensor(2)
    t3 = Tensor(-1.0f0)

    # each tensor can be multiplied by a scalar (tensor) from both sides
    @test t1 * t2 == t2 * t1 == 2 * t1 == t1 * 2 == Tensor(t1.data * 2)
    @test t1 * t3 == t3 * t1 == -1.0f0 * t1 == t1 * -1.0f0 == Tensor(t1.data * -1.0f0)

    # each tensor can be divided by a scalar (tensor)
    @test t1 / t2 == t1 / 2 == Tensor(t1.data / 2)
    @test t1 / t3 == t1 / -1.0f0 == Tensor(t1.data / -1.0f0)

    # non-scalar tensors can not be divided by
    @test_throws MethodError t2 / t1
    @test_throws MethodError -1.0f0 / t1

    # similar to Julia arrays only * and / are directly supported by tensors all other
    # operations need to Broadcasted
    @test_throws MethodError t1 ÷ t2
    @test_throws MethodError t1 ÷ 2
    @test_throws MethodError t1 \ t2
    @test_throws MethodError t1 \ 2
    @test_throws MethodError t1^t2
    @test_throws MethodError t1^2
    @test_throws MethodError t1 % t2
    @test_throws MethodError t1 % 2

    # this works
    @test t1 .÷ t2 == t1 .÷ 2 == Tensor(t1.data .÷ 2)
    @test t1 .\ t2 == t1 .\ 2 == Tensor(t1.data .\ 2)
    @test t1 .^ t2 == t1 .^ 2 == Tensor(t1.data .^ 2)
    @test t1 .% t2 == t1 .% 2 == Tensor(t1.data .% 2)
end
