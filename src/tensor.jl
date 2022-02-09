using ArgCheck

"""
    Tensor{T<:Number,N}

Wrapper type for native Julia arrays. Provides most of the standard array functionality and
extends it by arithmetic and algebraic tensor operations.

# Examples
```jldoctest
julia> t1 = Tensor([1 2; 3 4])
2×2 Tensor{Int64, 2}:
 1  2
 3  4

julia> 3.14 * t1
2×2 Tensor{Float64, 2}:
 3.14   6.28
 9.42  12.56

julia> t2 = Tensor(Float32[1.1, -2.2, 3.3])
3-element Tensor{Float32, 1}:
  1.1
 -2.2
  3.3

julia> t1 ⊗ t2
2×2×3 Tensor{Float32, 3}:
[:, :, 1] =
 1.1  2.2
 3.3  4.4

[:, :, 2] =
 -2.2  -4.4
 -6.6  -8.8

[:, :, 3] =
 3.3   6.6
 9.9  13.2
```

julia> foo(t1)
10.0
"""
struct Tensor{T<:Number,N}
    data::AbstractArray{T,N}
    Tensor(arr::AbstractArray{T,N}) where {T,N} = new{T,N}(arr)
    Tensor(val::T) where {T} = new{T,0}(fill(val))
end

@noinline function foo(t::Tensor)
    nt = convert(Tensor{Float64}, t)
    sum(nt.data)
end

# TODO: Add tensor product examples to documentation once implemented

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
Base.reshape(t::Tensor, dims...) = Tensor(reshape(t.data, dims...))
Base.permutedims(t::Tensor{T,1}) where {T} = Tensor(permutedims(t.data))
Base.permutedims(t::Tensor{T,2}) where {T} = Tensor(permutedims(t.data))
Base.permutedims(t::Tensor, perm) = Tensor(permutedims(t.data, perm))

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

Base.axes(t::Tensor, inds...) = axes(t.data, inds...)
Base.eachindex(t::Tensor) = eachindex(t.data)
Base.CartesianIndices(t::Tensor) = CartesianIndices(t.data)

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

# tensor hash should be slightly different than the hash of the underlying data
Base.hash(t::Tensor) = hash(hash(t.data) + hash("t"))
