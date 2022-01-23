using ArgCheck

"""
Awesome `Tensor` data type!
"""
struct Tensor{T<:Number, N}
    data::Array{T, N}
    Tensor(arr::Array{T, N}) where {T, N} = new{T, N}(arr)
    Tensor(val::T) where T = new{T, 0}(fill(val))
end

# Extend array property methods to tensors
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)
Base.eltype(t::Tensor) = eltype(t.data)

# Allow indexing tensors like arrays
function Base.getindex(t::Tensor, args...; kwargs...)
    @argcheck ndims(t) > 0 BoundsError(t)
    val = getindex(t.data, args...; kwargs...)
    Tensor(typeof(val) <: AbstractArray ? val : fill(val))
end

Base.setindex!(t::Tensor, val::Tensor, args...; kwargs...) = setindex!(t.data, val.data, args...; kwargs...)
Base.setindex!(t::Tensor, val::T, args...; kwargs...) where {T<:Number} = setindex!(t.data, val, args...; kwargs...)

Base.axes(t::Tensor, args...; kwargs...) = axes(t.data, args...; kwargs...)
Base.eachindex(t::Tensor, args...; kwargs...) = eachindex(t.data, args...; kwargs...)