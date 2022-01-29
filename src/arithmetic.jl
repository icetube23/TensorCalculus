using ArgCheck

# arithmetic comparisons between scalar tensors and scalars
Base.:<(t::Tensor{T,0}, val::S) where {T,S<:Number} = t.data[1] < val
Base.:<(val::S, t::Tensor{T,0}) where {T,S<:Number} = val < t.data[1]
Base.:(==)(t::Tensor{T,0}, val::S) where {T,S<:Number} = t.data[1] == val
Base.:(==)(val::S, t::Tensor{T,0}) where {T,S<:Number} = val == t.data[1]

# arithmetic comparisons between scalar tensors
Base.:<(t1::Tensor{T,0}, t2::Tensor{S,0}) where {T,S} = t1.data[1] < t2.data[1]

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
