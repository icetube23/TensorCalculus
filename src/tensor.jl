using ArgCheck

"""
Awesome `Tensor` data type!
"""
struct Tensor{T<:Number, N}
    data::Array{T, N}
    Tensor(arr::Array{T, N}) where {T, N} = new{T, N}(arr)
    Tensor(val::T) where T = new{T, 0}(fill(val))
end

# TODO: Add reshape and permutedims methods
# TODO: Add show methods
# TODO: Customize tensor broadcasting

# Extend array property methods to tensors
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)
Base.length(t::Tensor) = length(t.data)
Base.eltype(t::Tensor) = eltype(t.data)

# Extent array equality to tensors
Base.:(==)(t1::Tensor, t2::Tensor) = t1.data == t2.data

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

# Base.show overloads
Base.show(io::IO, t::Tensor{T, 0}) where T = print(io, "$(typeof(t))($(t.data[1]))")
function Base.show(io::IO, ::MIME"text/plain", t::Tensor)
    dims = ndims(t)
    shape = dims == 0 ? "scalar" :
            dims == 1 ? "$(length(t))-element" : join(size(t), "×")
    print(io, "$shape $(typeof(t)):\n")
    tmp_io = IOBuffer()
    show(tmp_io, "text/plain", t.data)
    data_str = String(take!(tmp_io))
    data_str = join(split(data_str, "\n")[2:end], "\n")
    print(io, data_str)
end