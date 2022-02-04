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

# TODO: Implement Base.hash
# TODO: Add proper documentation for Tensor
# TODO: Implement item() function for scalar tensors?

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
function Base.getindex(t::Tensor, inds...)
    @argcheck ndims(t) > 0 BoundsError(t)
    checkbounds(t, inds...)
    val = getindex(t.data, inds...)
    return Tensor(typeof(val) <: AbstractArray ? val : fill(val))
end

function Base.setindex!(t1::Tensor, t2::Tensor, inds...)
    checkbounds(t1, inds...)
    val = ndims(t2) == 0 ? t2.data[1] : t2.data
    return setindex!(t1.data, val, inds...)
end

function Base.setindex!(t::Tensor, val::S, inds...) where {S<:Number}
    checkbounds(t, inds...)
    return setindex!(t.data, val, inds...)
end

function Base.checkbounds(::Type{Bool}, t::Tensor, inds...)
    return checkbounds(Bool, t.data, inds...)
end

function Base.checkbounds(t::Tensor, inds...)
    if !checkbounds(Bool, t::Tensor, inds...)
        throw(BoundsError(t, inds))
    end
end

Base.axes(t::Tensor, args...; kwargs...) = axes(t.data, args...; kwargs...)
Base.eachindex(t::Tensor, args...; kwargs...) = eachindex(t.data, args...; kwargs...)

# Base.show overloads
Base.show(io::IO, t::Tensor{T,0}) where {T} = print(io, "$(typeof(t))($(t.data[1]))")
# TODO: Add ellipsis handling for large arrays
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
