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
- `ArgumentError`: if either factor is 0-dimensional or the last dimension of `t1` does not
    match the first dimension of `t2`
"""
function ⋅(t1::Tensor{T}, t2::Tensor{S}) where {T,S}
    @argcheck ndims(t1) > 0
    @argcheck ndims(t2) > 0
    @argcheck last(size(t1)) == first(size(t2))

    res = Array{promote_type(T, S)}(undef, front(size(t1))..., tail(size(t2))...)
    inds = CartesianIndices(res)

    # allow more efficient access of the needed (column-major) array elements
    a1 = permutedims(t1.data, (ndims(t1), 1:(ndims(t1) - 1)...))
    a2 = t2.data

    for i in eachindex(res)
        # map indices of result array to appropriate indices for the factor arrays
        ind = Tuple(inds[i])
        i1, i2 = ind[firstindex(ind):(ndims(t1) - 1)], ind[ndims(t1):lastindex(ind)]

        @inbounds res[i] = sum(a1[j, i1...] * a2[j, i2...] for j in axes(a1, 1))
    end

    return Tensor(res)
end

"""
    contract(t::Tensor{T}, i::Integer, j::Integer) where {T} -> Tensor{T}

Computes the contraction of the tensor `t` along the dimensions `i` and `j`.
`i` and `j` need to be distinct, valid indices for the dimensions of `t`. Furthermore, the
`i`-th dimension of `t` has to match its `j`-th dimension.

# Arguments
- `t::Tensor{T}`: the tensor to be contracted
- `i::Integer`: first contraction dimension
- `j::Integer`: second contraction dimension

# Returns
- `Tensor`: the contracted tensor

# Throws
- `ArgumentError`: if `i` and `j` are not distinct
- `BoundsError`: if `i` and `j` do not index valid dimensions of `t`
- `DimensionMismatch`: if the `i`-th dimension does not match the `j`-th dimension of `t`
"""
function contract(t::Tensor{T}, i::Integer, j::Integer) where {T}
    @argcheck i != j
    @argcheck 1 <= i <= ndims(t) BoundsError(size(t), i)
    @argcheck 1 <= j <= ndims(t) BoundsError(size(t), j)
    @argcheck size(t, i) == size(t, j) DimensionMismatch(
        "i-th dimension is $(size(t, i)) but j-th dimension is $(size(t, j))"
    )

    i, j = (min(i, j), max(i, j))
    st = size(t)
    res = Array{T}(
        undef,
        st[firstindex(st):(i - 1)]...,
        st[(i + 1):(j - 1)]...,
        st[(j + 1):lastindex(st)]...,
    )
    inds = CartesianIndices(res)

    # allow more efficient access of the needed (column-major) array elements
    a = permutedims(t.data, (i, j, Tuple(k for k in 1:ndims(t) if k != i && k != j)...))

    for n in eachindex(res)
        @inbounds res[n] = sum(a[m, m, inds[n]] for m in axes(a, 1))
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

# TODO: Implement epsilon tensor
# TODO: Implement delta tensor
# TODO: Think about sparse tensors for efficiency?
# TODO: Implement cross product (how?)
# TODO: Implement (repeated) 'Überschiebung'
# TODO: Add examples to all relevant docstrings
