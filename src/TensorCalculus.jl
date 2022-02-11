module TensorCalculus

export Tensor, ⊗, ⋅, contract, trace

include("tensor.jl")
include("broadcast.jl")
include("arithmetic.jl")
include("algebra.jl")

end
