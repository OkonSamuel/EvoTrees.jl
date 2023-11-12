function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:GradientRegression,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:LogLoss,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] = clamp(pred[1, i] + tree.pred[1, nid], T(-15), T(15))
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:MLE2P,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds pred[1, i] += tree.pred[1, nid]
        @inbounds pred[2, i] = max(T(-15), pred[2, i] + tree.pred[2, nid])
    end
    return nothing
end

function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L<:MLogLoss,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
        @views pred[:, i] .= max.(T(-15), pred[:, i] .- maximum(pred[:, i]))
    end
    return nothing
end

"""
    predict!(pred::Matrix, tree::Tree, X)

Generic fallback to add predictions of `tree` to existing `pred` matrix.
"""
function predict!(pred::Matrix{T}, tree::Tree{L,K}, x_bin::Matrix{UInt8}, feattypes::Vector{Bool}) where {L,K,T}
    @threads for i in axes(x_bin, 1)
        nid = 1
        @inbounds while tree.split[nid]
            feat = tree.feat[nid]
            cond = feattypes[feat] ? x_bin[i, feat] <= tree.cond_bin[nid] : x_bin[i, feat] == tree.cond_bin[nid]
            nid = nid << 1 + !cond
        end
        @inbounds for k = 1:K
            pred[k, i] += tree.pred[k, nid]
        end
    end
    return nothing
end

"""
    predict(model::EvoTree, X::AbstractMatrix; ntree_limit = length(model.trees))

Predictions from an EvoTree model - sums the predictions from all trees composing the model.
Use `ntree_limit=N` to only predict with the first `N` trees.
"""
function predict(
    m::EvoTree,
    data,
    ::Type{<:Device}=CPU;
    ntree_limit=length(m.params.trees))

    Tables.istable(data) ? data = Tables.columntable(data) : nothing
    trees = m.params.trees
    info = m.params.info
    L = m.params.loss_type
    K = m.params.outsize
    ntrees = length(trees)
    ntree_limit > ntrees && error("ntree_limit is larger than number of trees $ntrees.")
    x_bin = binarize(data; fnames=info[:fnames], edges=info[:edges])
    nobs = size(x_bin, 1)
    pred = zeros(Float32, K, nobs)
    for i = 1:ntree_limit
        predict!(pred, trees[i], x_bin, info[:feattypes])
    end
    if L == LogLoss
        pred .= sigmoid.(pred)
    elseif L ∈ [Poisson, Gamma, Tweedie]
        pred .= exp.(pred)
    elseif L in [GaussianMLE, LogisticMLE]
        pred[2, :] .= exp.(pred[2, :])
    elseif L == MLogLoss
        softmax!(pred)
    end
    pred = K == 1 ? vec(Array(pred')) : Array(pred')
    return pred
end

function softmax!(p::AbstractMatrix)
    @threads for i in axes(p, 2)
        _p = view(p, :, i)
        _p .= exp.(_p)
        isum = sum(_p)
        _p ./= isum
    end
    return nothing
end

function pred_leaf_cpu!(p::AbstractMatrix, n, ∑::AbstractVector, ::Type{<:GradientRegression}, config)
    ϵ = eps(eltype(p))
    p[1, n] = -config.eta * ∑[1] / max(ϵ, (∑[2] + config.lambda * ∑[3] + config.L2))
end
function pred_scalar(∑::AbstractVector, ::Type{<:GradientRegression}, config)
    ϵ = eps(eltype(∑))
    -config.eta * ∑[1] / max(ϵ, (∑[2] + config.lambda * ∑[3] + config.L2))
end


# prediction in Leaf - MLE2P
function pred_leaf_cpu!(p::AbstractMatrix, n, ∑::AbstractVector, ::Type{<:MLE2P}, config)
    ϵ = eps(eltype(p))
    p[1, n] = -config.eta * ∑[1] / max(ϵ, (∑[3] + config.lambda * ∑[5] + config.L2))
    p[2, n] = -config.eta * ∑[2] / max(ϵ, (∑[4] + config.lambda * ∑[5] + config.L2))
end
function pred_scalar(∑::AbstractVector, ::Type{<:MLE2P}, config)
    ϵ = eps(eltype(∑))
    -config.eta * ∑[1] / max(ϵ, (∑[3] + config.lambda * ∑[5] + config.L2))
end

# prediction in Leaf - MultiClassRegression
function pred_leaf_cpu!(p::AbstractMatrix, n, ∑::AbstractVector, ::Type{MLogLoss}, config)
    ϵ = eps(eltype(p))
    K = size(p, 1)
    @inbounds for k = axes(p, 1)
        p[k, n] = -config.eta * ∑[k] / max(ϵ, (∑[k+K] + config.lambda * ∑[end] + config.L2))
    end
end

# prediction in Leaf - L1
function pred_leaf_cpu!(p::AbstractMatrix, n, ∑::AbstractVector, ::Type{L1}, config)
    ϵ = eps(eltype(p))
    p[1, n] = config.eta * ∑[1] / max(ϵ, (∑[3] * (1 + config.lambda + config.L2)))
end
function pred_scalar(∑::AbstractVector, ::Type{L1}, config)
    ϵ = eps(eltype(∑))
    config.eta * ∑[1] / max(ϵ, (∑[3] * (1 + config.lambda + config.L2)))
end
