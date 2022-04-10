using ArgCheck

# arithmetic comparisons between scalar tensors and scalars
Base.isless(t::Tensor{T,0}, val::S) where {T,S<:Number} = t.data[1] < val
Base.isless(val::S, t::Tensor{T,0}) where {T,S<:Number} = val < t.data[1]
Base.:(==)(t::Tensor{T,0}, val::S) where {T,S<:Number} = t.data[1] == val
Base.:(==)(val::S, t::Tensor{T,0}) where {T,S<:Number} = val == t.data[1]

# arithmetic comparisons between scalar tensors
Base.isless(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = t1.data[1] < t2.data[1]

# arithmetic operations between scalar tensors and scalars
Base.:+(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] + val)
Base.:+(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val + t.data[1])
Base.:-(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] - val)
Base.:-(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val - t.data[1])
Base.:*(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] * val)
Base.:*(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val * t.data[1])
Base.:/(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] / val)
Base.:/(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val / t.data[1])
Base.:÷(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] ÷ val)
Base.:÷(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val ÷ t.data[1])
Base.:\(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] \ val)
Base.:\(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val \ t.data[1])
Base.:^(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1]^val)
Base.:^(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val^t.data[1])
Base.:%(t::Tensor{T,0}, val::S) where {T,S<:Number} = Tensor(t.data[1] % val)
Base.:%(val::S, t::Tensor{T,0}) where {T,S<:Number} = Tensor(val % t.data[1])

# arithmetic operations between scalar tensors
Base.:*(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1] * t2.data[1])
Base.:/(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1] / t2.data[1])
Base.:÷(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1] ÷ t2.data[1])
Base.:\(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1] \ t2.data[1])
Base.:^(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1]^t2.data[1])
Base.:%(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = Tensor(t1.data[1] % t2.data[1])

# arithmetic operations for arbitrary tensors
Base.:+(t::Tensor) = t
function Base.:+(t1::Tensor, t2::Tensor)
    @argcheck size(t1) == size(t2) DimensionMismatch
    return Tensor(t1.data + t2.data)
end

Base.:-(t::Tensor) = Tensor(-t.data)
function Base.:-(t1::Tensor, t2::Tensor)
    @argcheck size(t1) == size(t2) DimensionMismatch
    return Tensor(t1.data - t2.data)
end

# NOTE: Scalar tensors behaving like scalars for these operations might
# be removed in future versions
Base.:*(t::Tensor, val::S) where {S<:Number} = Tensor(t.data * val)
Base.:*(val::S, t::Tensor) where {S<:Number} = Tensor(val * t.data)
Base.:*(t1::Tensor, t2::Tensor{T,0}) where {T} = Tensor(t1.data * t2.data[1])
Base.:*(t1::Tensor{T,0}, t2::Tensor) where {T} = Tensor(t1.data[1] * t2.data)

Base.:/(t::Tensor, val::S) where {S<:Number} = Tensor(t.data / val)
Base.:/(t1::Tensor, t2::Tensor{T,0}) where {T} = Tensor(t1.data / t2.data[1])

# allow typical array operations like sum or maximum for tensors
function Base.sum(f::Union{Function,Type}, t::Tensor; kwargs...)
    return Tensor(sum(f, t.data; kwargs...))
end
Base.sum(t::Tensor; kwargs...) = sum(identity, t; kwargs...)

function Base.prod(f::Union{Function,Type}, t::Tensor; kwargs...)
    return Tensor(prod(f, t.data; kwargs...))
end
Base.prod(t::Tensor; kwargs...) = prod(identity, t; kwargs...)

function Base.maximum(f::Union{Function,Type}, t::Tensor; kwargs...)
    return Tensor(maximum(f, t.data; kwargs...))
end
Base.maximum(t::Tensor; kwargs...) = maximum(identity, t; kwargs...)

function Base.minimum(f::Union{Function,Type}, t::Tensor; kwargs...)
    return Tensor(minimum(f, t.data; kwargs...))
end
Base.minimum(t::Tensor; kwargs...) = minimum(identity, t; kwargs...)

function Base.extrema(f::Union{Function,Type}, t::Tensor; kwargs...)
    return (minimum(f, t; kwargs...), maximum(f, t; kwargs...))
end
Base.extrema(t::Tensor; kwargs...) = extrema(identity, t; kwargs...)

Base.argmax(t::Tensor; kwargs...) = argmax(t.data; kwargs...)
Base.argmin(t::Tensor; kwargs...) = argmin(t.data; kwargs...)

Base.findmax(t::Tensor; kwargs...) = (maximum(t; kwargs...), argmax(t; kwargs...))
Base.findmin(t::Tensor; kwargs...) = (minimum(t; kwargs...), argmin(t; kwargs...))

Base.conj(t::Tensor) = Tensor(conj(t.data))
