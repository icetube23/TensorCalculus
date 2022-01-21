struct Tensor{T, N}
    data::Array{T, N}
end

Base.ndims(t::Tensor) = ndims(t.data)
Base.size(t::Tensor) = size(t.data)
Base.eltype(t::Tensor) = eltype(t.data)