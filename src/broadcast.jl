"""
    BroadcastTensor{T,N} <: AbstractArray{T,N}

An efficient broadcasting wrapper for `Tensor`.
Sub-types `AbstractArray` to make use of existing broadcasting utilities.
"""
struct BroadcastTensor{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
    BroadcastTensor(t::Tensor{T,N}) where {T,N} = new{T,N}(t.data)
end

# behaves like getindex(::Tensor, ...) but returns scalars instead of scalar tensors
function Base.getindex(t::BroadcastTensor, inds...)
    val = getindex(t.data, inds...)
    return typeof(val) <: AbstractArray ? Tensor(val) : val
end

# needed to allow broadcasting a BroadcastTensor
Base.axes(t::BroadcastTensor, inds...) = axes(t.data, inds...)

# define the broadcasting behaviour of tensors using BroadcastTensor
Base.broadcastable(t::Tensor) = BroadcastTensor(t)
Base.BroadcastStyle(::Type{<:Tensor}) = Broadcast.ArrayStyle{BroadcastTensor}()
Base.BroadcastStyle(::Type{<:BroadcastTensor}) = Broadcast.ArrayStyle{BroadcastTensor}()

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{BroadcastTensor}}, ::Type{ElType}
) where {ElType}
    return Tensor(similar(Array{ElType}, axes(bc)))
end

Base.copyto!(t::Tensor, bc::Broadcast.Broadcasted) = Tensor(copyto!(t.data, bc))
