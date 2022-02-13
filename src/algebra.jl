using Base: front, tail

"""
    ⊗(t1::Tensor{T}, t2::Tensor{S}) where {T,S} -> Tensor{promote_type(T, S)}

Computes the outer product of the tensors `t1` and `t2`, also referred to as the tensor
product of `t1` and `t2`.
Arguments can be of arbitrary size and dimension.

# Arguments
- `t1::Tensor{T}`: the first factor
- `t2::Tensor{S}`: the second factor

# Returns
- `Tensor{promote_type(T, S)}`: the outer product of `t1` and `t2`
"""
function ⊗(t1::Tensor{T}, t2::Tensor{S}) where {T,S}
    res = Array{promote_type(T, S)}(undef, size(t1)..., size(t2)...)
    inds1, inds2 = CartesianIndices(t1), CartesianIndices(t2)

    # FIXME: way to many allocs here, probably bad index usage
    for i in eachindex(t1), j in eachindex(t2)
        @inbounds res[inds1[i], inds2[j]] = t1.data[i] * t2.data[j]
    end

    return Tensor(res)
end

"""
    ⊗(ts...) -> Tensor{promote_type(eltype.(ts)...)}

Computes the outer product of an arbitrary number of tensors `ts...`. The empty outer
product is defined as `Tensor(1)` (i.e., the neutral element of the outer product).
Arguments can be of arbitrary size and dimension.

# Arguments
- `ts...`: the (possibly empty) list of tensors to be multiplied

# Returns
- `Tensor{promote_type(eltype.(ts)...)}`: the outer product of the tensors `ts...`
"""
⊗(ts...) = reduce(⊗, ts; init=Tensor(1))

"""
    ⋅(t1::Tensor{T}, t2::Tensor{S}) where {T,S} -> Tensor{promote_type(T, S)}

Computes the inner product of the tensors `t1` and `t2`, also referred to as the scalar
product of `t1` and `t2`.
Arguments need to be of non-zero dimension and the last dimension of `t1` needs to match the
first dimension of `t2`.

# Arguments
- `t1::Tensor{T}`: the first factor
- `t2::Tensor{S}`: the second factor

# Returns
- `Tensor{promote_type(T, S)}`: the inner product of `t1` and `t2`

# Throws
- `ArgumentError`: if either `t1` or `t2` is 0-dimensional
- `DimensionMismatch`: if the last dimension of `t1` does not match the first dimension of
    `t2`
"""
function ⋅(t1::Tensor{T}, t2::Tensor{S}) where {T,S}
    # TODO: benchmark inner product against pushover
    @argcheck ndims(t1) > 0
    @argcheck ndims(t2) > 0
    @argcheck last(size(t1)) == first(size(t2)) DimensionMismatch

    res = Array{promote_type(T, S)}(undef, front(size(t1))..., tail(size(t2))...)
    inds = CartesianIndices(res)

    # allow more efficient access of the needed (column-major) array elements
    a1 = permutedims(conj(t1.data), (ndims(t1), 1:(ndims(t1) - 1)...))
    a2 = t2.data

    # FIXME: too many allocs, probably inefficient indexing
    for i in eachindex(res)
        # map indices of result array to appropriate indices for the factor arrays
        ind = Tuple(inds[i])
        i1, i2 = ind[firstindex(ind):(ndims(t1) - 1)], ind[ndims(t1):lastindex(ind)]

        @inbounds res[i] = sum(a1[j, i1...] * a2[j, i2...] for j in axes(a1, 1))
    end

    return Tensor(res)
end

"""
    contract(t::Tensor, d1::Integer, d2::Integer) -> Tensor

Computes the contraction of the tensor `t` along the dimensions `d1` and `d2`.
`d1` and `d2` need to be distinct, valid indices for the dimensions of `t`. Additionally,
the `d1`-th dimension of `t` has to match its `d2`-th dimension.

# Arguments
- `t::Tensor`: the tensor to be contracted
- `d1::Integer`: the first contraction dimension
- `d2::Integer`: the second contraction dimension

# Returns
- `Tensor`: the contracted tensor

# Throws
- `ArgumentError`: if `d1` and `d2` are not distinct
- `BoundsError`: if `d1` and `d2` do not index valid dimensions of `t`
- `DimensionMismatch`: if the `d1`-th dimension does not match the `d2`-th dimension of `t`
"""
function contract(t::Tensor, d1::Integer, d2::Integer)
    @argcheck d1 != d2
    @argcheck 1 <= d1 <= ndims(t) BoundsError(size(t), d1)
    @argcheck 1 <= d2 <= ndims(t) BoundsError(size(t), d2)
    @argcheck size(t, d1) == size(t, d2) DimensionMismatch

    d1, d2 = (min(d1, d2), max(d1, d2))
    st = size(t)
    res = Array{eltype(t)}(
        undef,
        st[firstindex(st):(d1 - 1)]...,
        st[(d1 + 1):(d2 - 1)]...,
        st[(d2 + 1):lastindex(st)]...,
    )
    inds = CartesianIndices(res)

    # allow more efficient access of the needed (column-major) array elements
    a = permutedims(t.data, (d1, d2, Tuple(k for k in 1:ndims(t) if k != d1 && k != d2)...))

    # FIXME: allocs are not too bad here, but probably can still remove most of them
    for i in eachindex(res)
        @inbounds res[i] = sum(a[j, j, inds[i]] for j in axes(a, 1))
    end

    return Tensor(res)
end

"""
    trace(t::Tensor{T,2}) where {T} -> Tensor{T,0}

Computes the contraction of the tensor `t` along its two dimensions, equivalent to the trace
of the underlying matrix of `t`.
The two dimensions `t` need to match, i.e., `t` needs to be a 'square' tensor.

# Arguments
- `t::Tensor{T,2}`: the tensor to be contracted

# Returns
- `Tensor{T,0}`: the contracted scalar tensor

# Throws
- `DimensionMismatch`: if the first and the second dimension of `t` do not match
"""
trace(t::Tensor{T,2}) where {T} = contract(t, 1, 2)

"""
    pushover(t1::Tensor{T}, t2::Tensor{S}, d1::Integer, d2::Integer) where {T,S} -> Tensor{promote_type(T, S)}

Computes the 'Überschiebung' of the tensors `t1` and `t2` along the dimensions `d1` and `d2`
respectively.
`d1` and `d2` need to be valid indices for the dimensions of `t1` and `t2` respectively.
Additionally, the `d1`-th dimension of `t1` has to match the `d2`-th dimension of `t2`.

# Arguments
- `t1::Tensor{T}`: the first tensor to push over
- `t2::Tensor{S}`: the second tensor to push over
- `d1::Integer`: the first push-over dimension
- `d2::Integer`: the second push-over dimension

# Returns
- `Tensor{promote_type(T, S)}`: the push-over tensor

# Throws
- `BoundsError`: if `d1` and `d2` do not index valid dimensions of `t1` and `t2`
    respectively
- `DimensionMismatch`: if the `d1`-th dimension of `t1` does not match the `d2`-th dimension
    of `t2`
"""
function pushover(t1::Tensor{T}, t2::Tensor{S}, d1::Integer, d2::Integer) where {T,S}
    # TODO: benchmark pushover against permutedims + inner product
    @argcheck 1 <= d1 <= ndims(t1) BoundsError(size(t1), d1)
    @argcheck 1 <= d2 <= ndims(t2) BoundsError(size(t2), d2)
    @argcheck size(t1, d1) == size(t2, d2) DimensionMismatch

    st1 = size(t1)
    st2 = size(t2)
    res = Array{promote_type(T, S)}(
        undef,
        st1[firstindex(st1):(d1 - 1)]...,
        st1[(d1 + 1):lastindex(st1)]...,
        st2[firstindex(st2):(d2 - 1)]...,
        st2[(d2 + 1):lastindex(st2)]...,
    )
    inds = CartesianIndices(res)

    # allow more efficient access of the needed (column-major) array elements
    a1 = permutedims(t1.data, (d1, Tuple(k for k in 1:ndims(t1) if k != d1)...))
    a2 = permutedims(t2.data, (d2, Tuple(k for k in 1:ndims(t2) if k != d2)...))

    # FIXME: also way too many allocs, improve indexing
    for i in eachindex(res)
        ind = Tuple(inds[i])
        i1, i2 = ind[firstindex(ind):(ndims(t1) - 1)], ind[ndims(t1):lastindex(ind)]

        @inbounds res[i] = sum(a1[j, i1...] * a2[j, i2...] for j in axes(a1, 1))
    end

    return Tensor(res)
end

# TODO: Implement epsilon tensor
# TODO: Implement delta tensor
# TODO: Think about sparse tensors for efficiency?
# TODO: Implement cross product (how?)
# TODO: Add examples to all relevant docstrings
# TODO: Maybe rename actual implementations of ⊗ and ⋅ to outer and inner and alias symbols
