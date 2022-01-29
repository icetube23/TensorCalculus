using TensorCalculus
using Test

@testset "TensorCalculus" begin
    @testset "Tensor" begin
        include("tensor.jl")
    end

    @testset "Broadcasting" begin
        include("broadcast.jl")
    end
end
