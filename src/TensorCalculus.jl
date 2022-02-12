module TensorCalculus

export Tensor, ⊗, ⋅, contract, trace, pushover

include("tensor.jl")
include("broadcast.jl")
include("arithmetic.jl")
include("algebra.jl")

end
