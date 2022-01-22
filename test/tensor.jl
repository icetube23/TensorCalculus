@testset "tensor.jl" begin
    t1 = Tensor(rand(3, 3))
    @test ndims(t1) == 2
    t2 = Tensor(rand(3, 3, 3))
    @test ndims(t2) == 3
end