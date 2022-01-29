struct BroadcastTensor
    data::Array
    BroadcastTensor(t::Tensor) = new(t.data)
end

# behaves like getindex(::Tensor, ...) but returns scalars instead of scalar tensors
function Base.getindex(t::BroadcastTensor, args...; kwargs...)
    val = getindex(t.data, args...; kwargs...)
    typeof(val) <: AbstractArray ? Tensor(val) : val
end

# needed to allow broadcasting a BroadcastTensor
Base.axes(t::BroadcastTensor, args...; kwargs...) = axes(t.data, args...; kwargs...)
Base.ndims(t::BroadcastTensor) = ndims(t.data)

# define the broadcasting behaviour of tensors using BroadcastTensor
Base.broadcastable(t::Tensor) = BroadcastTensor(t)
Base.BroadcastStyle(::Type{<:Tensor}) = Broadcast.Style{Tensor}()
Base.BroadcastStyle(::Type{<:BroadcastTensor}) = Broadcast.Style{Tensor}()
Base.BroadcastStyle(::Broadcast.Style{Tensor}, ::Broadcast.BroadcastStyle) = Broadcast.Style{Tensor}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.Style{Tensor}}, ::Type{ElType}) where ElType = Tensor(similar(Array{ElType}, axes(bc)))
Base.copyto!(t::Tensor, bc::Broadcast.Broadcasted) = Tensor(copyto!(t.data, bc))
