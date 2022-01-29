@testset "Implicit broadcasting" begin
    # test that tensor broadcasting behaves exactly like array broadcasting
    a1 = reshape(Vector(1:9), 3, 3)
    @test Tensor(a1) .^ 2 == Tensor(a1 .^ 2)

    a2 = rand(3, 3, 3)
    @test (Tensor(a2) .< 0.5) == Tensor(a2 .< 0.5)

    a3 = rand(3, 3, 3)
    @test (Tensor(a2) .< Tensor(a3)) == Tensor(a2 .< a3)

    a4 = randn(4, 2, 5, 3)
    @test abs.(Tensor(a4)) == Tensor(abs.(a4))

    # more complex broadcasting example taken from a blog post at
    # https://julialang.org/blog/2018/05/extensible-broadcast-fusion/
    a5 = [1, 2, 3]
    a6 = [10 20 30 40]
    a7 = 10

    # test broadcasting behaviour of the expression (a5 .+ a6) ./ a7
    @test (Tensor(a5) .+ a6) ./ a7 == Tensor((a5 .+ a6) ./ a7)
    @test (a5 .+ Tensor(a6)) ./ a7 == Tensor((a5 .+ a6) ./ a7)
    @test (a5 .+ a6) ./ Tensor(a7) == Tensor((a5 .+ a6) ./ a7)
    @test (Tensor(a5 .+ a6)) ./ a7 == Tensor((a5 .+ a6) ./ a7)

    @test (Tensor(a5 .+ a6)) ./ Tensor(a7) == Tensor((a5 .+ a6) ./ a7)
    @test (Tensor(a5) .+ a6) ./ Tensor(a7) == Tensor((a5 .+ a6) ./ a7)
    @test (a5 .+ Tensor(a6)) ./ Tensor(a7) == Tensor((a5 .+ a6) ./ a7)
    @test (Tensor(a5) .+ Tensor(a6)) ./ a7 == Tensor((a5 .+ a6) ./ a7)

    @test (Tensor(a5) .+ Tensor(a6)) ./ Tensor(a7) == Tensor((a5 .+ a6) ./ a7)
end

@testset "Explicit broadcasting" begin
    a1 = [1 2 3 4 5]
    a2 = [1 3 5 7 9;
          2 4 6 8 10]
    t1 = Tensor(a1)
    t2 = Tensor(a2)

    # explicit broadcasting works as expected
    @test broadcast(+, t1, a2) == Tensor([2 5 8 11 14;
                                          3 6 9 12 15])
    @test broadcast(+, a1, t2) == Tensor([2 5 8 11 14;
                                          3 6 9 12 15])
    @test broadcast(+, t1, t2) == Tensor([2 5 8 11 14;
                                          3 6 9 12 15])

    # explicit in-place broadcasting is also possible
    @test t2 != Tensor([2 5 8 11 14;
                        3 6 9 12 15])
    broadcast!(+, t2, t1, t2)
    @test t2 == Tensor([2 5 8 11 14;
                        3 6 9 12 15])
end
