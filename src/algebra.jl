"""
    ⊗(t1::Tensor{T}, t2::Tensor{S}) where {T,S} -> Tensor{promote_type(T,S)}

Computes the outer tensor product of the tensors `t1` and `t2`. `t1` and `t2` can be of
arbitrary size and dimension.

# Arguments
- `t1::Tensor{T}`: the first factor
- `t2::Tensor{S}`: the second factor
"""
function ⊗(t1::Tensor{T}, t2::Tensor{S}) where {T,S}
    res = Array{promote_type(T, S)}(undef, size(t1)..., size(t2)...)
    inds1, inds2 = CartesianIndices(t1), CartesianIndices(t2)
    for i in eachindex(t1), j in eachindex(t2)
        @inbounds res[inds1[i], inds2[j]] = t1.data[i] * t2.data[j]
    end
    return Tensor(res)
end
