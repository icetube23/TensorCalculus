using ArgCheck

"""
Awesome `Tensor` data type!
"""
struct Tensor{T<:Number,N}
    data::Array{T,N}
    Tensor(arr::Array{T,N}) where {T,N} = new{T,N}(arr)
    Tensor(arr::BitArray{N}) where {N} = new{Bool,N}(Array{Bool}(arr))
    Tensor(val::T) where {T} = new{T,0}(fill(val))
end

# extend array property methods to tensors
Base.size(t::Tensor) = size(t.data)
Base.ndims(t::Tensor) = ndims(t.data)
Base.length(t::Tensor) = length(t.data)
Base.eltype(t::Tensor) = eltype(t.data)

# extent array equality to tensors
Base.:(==)(t1::Tensor, t2::Tensor) = t1.data == t2.data
function Base.isapprox(t1::Tensor, t2::Tensor, args...; kwargs...)
    return isapprox(t1.data, t2.data, args...; kwargs...)
end

# type conversions
(::Type{Tensor{T}})(t::Tensor) where {T} = convert(Tensor{T}, t)
Base.convert(::Type{Tensor{T}}, t::Tensor) where {T} = Tensor(convert(Array{T}, t.data))
(::Type{T})(t::Tensor{T,0}) where {T} = t.data[1]
Base.convert(::Type{T}, t::Tensor{T,0}) where {T} = t.data[1]

# extend array size manipulation methods to tensors
Base.reshape(t::Tensor, args...; kwargs...) = Tensor(reshape(t.data, args...; kwargs...))
function Base.permutedims(t::Tensor, args...; kwargs...)
    return Tensor(permutedims(t.data, args...; kwargs...))
end

# allow indexing tensors like arrays
function Base.getindex(t::Tensor, args...; kwargs...)
    @argcheck ndims(t) > 0 BoundsError(t)
    val = getindex(t.data, args...; kwargs...)
    return Tensor(typeof(val) <: AbstractArray ? val : fill(val))
end

function Base.setindex!(t::Tensor, val::Tensor, args...; kwargs...)
    return setindex!(t.data, val.data, args...; kwargs...)
end

function Base.setindex!(t::Tensor, val::S, args...; kwargs...) where {S<:Number}
    return setindex!(t.data, val, args...; kwargs...)
end

Base.axes(t::Tensor, args...; kwargs...) = axes(t.data, args...; kwargs...)
Base.eachindex(t::Tensor, args...; kwargs...) = eachindex(t.data, args...; kwargs...)

# Base.show overloads
Base.show(io::IO, t::Tensor{T,0}) where {T} = print(io, "$(typeof(t))($(t.data[1]))")
function Base.show(io::IO, ::MIME"text/plain", t::Tensor)
    dims = ndims(t)
    shape = if dims == 0
        "scalar"
    elseif dims == 1
        "$(length(t))-element"
    else
        join(size(t), "×")
    end
    print(io, "$shape $(typeof(t)):\n")
    tmp_io = IOBuffer()
    show(tmp_io, "text/plain", t.data)
    data_str = String(take!(tmp_io))
    data_str = join(split(data_str, "\n")[2:end], "\n")
    return print(io, data_str)
end
