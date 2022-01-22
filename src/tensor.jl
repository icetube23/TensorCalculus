"""
Awesome `Tensor` data type!
"""
struct Tensor{T, N}
    data::Array{T, N}
end

Base.getindex(t::Tensor, args...; kwargs...) = getindex(t.data, args...; kwargs...)
Base.setindex!(t::Tensor, args...; kwargs...) = setindex!(t.data, args...; kwargs...)
Base.axes(t::Tensor, args...; kwargs...) = axes(t.data, args...; kwargs...)
Base.ndims(t::Tensor) = ndims(t.data)
Base.size(t::Tensor) = size(t.data)
Base.eltype(t::Tensor) = eltype(t.data)