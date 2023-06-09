function init_gpu(
    params::EvoTypes{L,T},
    dtrain::AbstractDataFrame;
    target_name,
    fnames=nothing,
    w_name=nothing,
    offset_name=nothing
) where {L,T}

    target_levels = nothing
    offset = !isnothing(offset_name) ? T.(dtrain[:, offset_name]) : nothing
    if L == Logistic
        K = 1
        y = T.(dtrain[!, target_name])
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        K = 1
        y = T.(dtrain[!, target_name])
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == Softmax
        if eltype(dtrain[!, target_name]) <: CategoricalValue
            target_levels = CategoricalArrays.levels(dtrain[:, target_name])
            y = UInt32.(CategoricalArrays.levelcode.(dtrain[:, target_name]))
        else
            target_levels = sort(unique(dtrain[!, target_name]))
            yc = CategoricalVector(dtrain[:, target_name], levels=target_levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        end
        K = length(target_levels)
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        K = 2
        y = T.(dtrain[!, target_name])
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        K = 2
        y = T.(dtrain[!, target_name])
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        K = 1
        y = T.(dtrain[!, target_name])
        μ = [mean(y)]
    end
    y = CuArray(y)
    μ = T.(μ)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    # initialize preds
    nobs = nrow(dtrain)
    pred = CUDA.zeros(T, K, nobs)
    pred .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # init EvoTree
    bias = [Tree{L,K,T}(μ)]

    # set fnames
    _w_name = isnothing(w_name) ? "" : string(w_name)
    _offset_name = isnothing(offset_name) ? "" : string(offset_name)
    if isnothing(fnames)
        fnames = String[]
        for name in names(dtrain)
            if eltype(dtrain[!, name]) <: Union{Real,CategoricalValue}
                push!(fnames, name)
            end
        end
        fnames = setdiff(fnames, union([target_name], [_w_name], [_offset_name]))
    else
        isa(fnames, String) ? fnames = [fnames] : nothing
        fnames = string.(fnames)
        @assert isa(fnames, Vector{String})
        for name in fnames
            @assert eltype(dtrain[!, name]) <: Union{Real,CategoricalValue}
        end
    end
    nfeats = length(fnames)

    # initialize gradients and weights
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    w = isnothing(w_name) ? CUDA.ones(T, size(y)) : CuVector{T}(dtrain[!, w_name])
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(dtrain; fnames, nbins=params.nbins, rng=params.rng)
    x_bin = CuArray(binarize(dtrain; fnames, edges))

    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(eltype(js_), ceil(Int, params.colsample * nfeats))

    # initialize histograms
    nodes = [TrainNode(featbins, K, view(is_in, 1:0), T) for n = 1:2^params.max_depth-1]
    # h∇ = [CUDA.zeros(T, 2 * K + 1, nbins) for nbins in featbins]
    h∇ = CUDA.zeros(T, 2 * K + 1, maximum(featbins), length(featbins))

    out = CUDA.zeros(UInt32, nobs)
    left = CUDA.zeros(UInt32, nobs)
    right = CUDA.zeros(UInt32, nobs)

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in params.monotone_constraints
        monotone_constraints[k] = v
    end

    info = Dict(
        :fnames => fnames,
        :target_name => target_name,
        :w_name => w_name,
        :offset_name => offset_name,
        :target_levels => target_levels,
        :edges => edges,
        :feattypes => feattypes,
    )

    # initialize model
    m = EvoTreeGPU{L,K,T}(bias, info)

    # store cache
    cache = (
        info=Dict(:nrounds => 0),
        x_bin=x_bin,
        y=y,
        w=w,
        K=K,
        nodes=nodes,
        pred=pred,
        is_in=is_in,
        is_out=is_out,
        mask=mask,
        js_=js_,
        js=js,
        out=out,
        left=left,
        right=right,
        ∇=∇,
        h∇=h∇,
        fnames=fnames,
        edges=edges,
        featbins=featbins,
        feattypes=feattypes,
        feattypes_gpu=CuArray(feattypes),
        monotone_constraints=monotone_constraints,
    )
    return m, cache
end


function train!(
    evotree::EvoTreeGPU{L,K,T},
    cache,
    params::EvoTypes{L,T},
) where {L,K,T}

    # compute gradients
    update_grads_gpu!(cache.∇, cache.pred, cache.y, params)
    # subsample rows
    cache.nodes[1].is =
        subsample_gpu(cache.is_in, cache.is_out, cache.mask, params.rowsample, params.rng)
    # subsample cols
    sample!(params.rng, cache.js_, cache.js, replace=false, ordered=true)

    # assign a root and grow tree
    tree = Tree{L,K,T}(params.max_depth)
    grow!(
        tree,
        cache.nodes,
        params,
        cache.∇,
        cache.edges,
        cache.js,
        cache.out,
        cache.left,
        cache.right,
        cache.h∇,
        cache.x_bin,
        cache.feattypes,
        cache.monotone_constraints,
    )
    push!(evotree.trees, tree)
    predict!(cache.pred, tree, cache.x_bin, cache.feattypes_gpu)
    cache[:info][:nrounds] += 1
    return nothing
end

# grow a single tree - grow through all depth
function grow!(
    tree::Tree{L,K,T},
    nodes::Vector{N},
    params::EvoTypes{L,T},
    ∇::CuMatrix{T},
    edges,
    js,
    out,
    left,
    right,
    h∇::CuArray{T,3},
    x_bin::CuMatrix,
    feattypes::Vector{Bool},
    monotone_constraints,
) where {L,K,T,N}

    jsg = CuVector(js)
    # reset nodes
    for n in nodes
        n.∑ .= 0
        n.gain = T(0)
        @inbounds for i in eachindex(n.h)
            n.h[i] .= 0
            n.gains[i] .= 0
        end
    end

    # initialize
    n_next = [1]
    n_current = copy(n_next)
    depth = 1

    # initialize summary stats
    nodes[1].∑ .= Vector(vec(sum(∇[:, nodes[1].is], dims=2)))
    nodes[1].gain = get_gain(params, nodes[1].∑) # should use a GPU version?
    # grow while there are remaining active nodes - TO DO histogram substraction hits issue on GPU
    while length(n_current) > 0 && depth <= params.max_depth
        offset = 0 # identifies breakpoint for each node set within a depth
        if depth < params.max_depth
            for n_id in eachindex(n_current)
                n = n_current[n_id]
                if n_id % 2 == 0
                    if n % 2 == 0
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n+1].h[j]
                        end
                    else
                        @inbounds for j in eachindex(nodes[n].h)
                            nodes[n].h[j] .= nodes[n>>1].h[j] .- nodes[n-1].h[j]
                        end
                    end
                else
                    # @info "hist"
                    update_hist_gpu!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, jsg, js)
                    # update_hist_gpu_vec!(nodes[n].h, h∇, ∇, x_bin, nodes[n].is, js)
                end
            end
        end

        # grow while there are remaining active nodes
        for n ∈ sort(n_current)
            if depth == params.max_depth || nodes[n].∑[end] <= params.min_weight
                pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
            else
                # @info "gain & max"
                update_gains!(nodes[n], js, params, feattypes, monotone_constraints)
                best = findmax(findmax.(nodes[n].gains))
                best_gain = best[1][1]
                best_bin = best[1][2]
                best_feat = best[2]
                # if best_gain > nodes[n].gain + params.gamma && best_gain > nodes[n].gains[best_feat][end] + params.gamma
                if best_gain > nodes[n].gain + params.gamma
                    tree.gain[n] = best_gain - nodes[n].gain
                    tree.cond_bin[n] = best_bin
                    tree.feat[n] = best_feat
                    tree.cond_float[n] = edges[tree.feat[n]][tree.cond_bin[n]]
                end
                tree.split[n] = tree.cond_bin[n] != 0
                if !tree.split[n]
                    pred_leaf_cpu!(tree.pred, n, nodes[n].∑, params, ∇, nodes[n].is)
                    popfirst!(n_next)
                else
                    # @info "split" best_bin typeof(nodes[n].is) length(nodes[n].is)
                    _left, _right = split_set_threads_gpu!(
                        out,
                        left,
                        right,
                        nodes[n].is,
                        x_bin,
                        tree.feat[n],
                        tree.cond_bin[n],
                        feattypes[best_feat],
                        offset,
                    )
                    offset += length(nodes[n].is)
                    nodes[n<<1].is, nodes[n<<1+1].is = _left, _right
                    nodes[n<<1].∑ .= nodes[n].hL[best_feat][:, best_bin]
                    nodes[n<<1+1].∑ .= nodes[n].hR[best_feat][:, best_bin]
                    nodes[n<<1].gain = get_gain(params, nodes[n<<1].∑)
                    nodes[n<<1+1].gain = get_gain(params, nodes[n<<1+1].∑)
                    if length(_right) >= length(_left)
                        push!(n_next, n << 1)
                        push!(n_next, n << 1 + 1)
                    else
                        push!(n_next, n << 1 + 1)
                        push!(n_next, n << 1)
                    end
                    # @info "split post" length(_left) length(_right)
                    popfirst!(n_next)
                end
            end
        end
        n_current = copy(n_next)
        depth += 1
    end # end of loop over active ids for a given depth

    return nothing
end