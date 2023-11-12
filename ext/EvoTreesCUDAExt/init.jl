function EvoTrees.init_core!(m::EvoTrees.EvoTree, ::Type{<:EvoTrees.GPU}, data, fnames, y, w, offset)

    config = m.config
    params = m.params
    cache = m.cache
    L = config.loss_type

    # binarize data into quantiles
    edges, featbins, feattypes = EvoTrees.get_edges(data; fnames, max_bins=config.max_bins, rng=config.rng)
    x_bin = CuArray(EvoTrees.binarize(data; fnames, edges))
    nobs, nfeats = size(x_bin)
    T = Float32

    target_levels = nothing
    if L == EvoTrees.Logistic
        @assert eltype(y) <: Real && minimum(y) >= 0 && maximum(y) <= 1
        K = 1
        y = T.(y)
        μ = [EvoTrees.logit(EvoTrees.mean(y))]
        !isnothing(offset) && (offset .= EvoTrees.logit.(offset))
    elseif L in [EvoTrees.Poisson, EvoTrees.Gamma, EvoTrees.Tweedie]
        @assert eltype(y) <: Real
        K = 1
        y = T.(y)
        μ = fill(log(EvoTrees.mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.MLogLoss
        if eltype(y) <: EvoTrees.CategoricalValue
            target_levels = EvoTrees.CategoricalArrays.levels(y)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(y))
        elseif eltype(y) <: Integer || eltype(y) <: Bool || eltype(y) <: String || eltype(y) <: Char
            target_levels = sort(unique(y))
            yc = EvoTrees.CategoricalVector(y, levels=target_levels)
            y = UInt32.(EvoTrees.CategoricalArrays.levelcode.(yc))
        else
            @error "Invalid target eltype: $(eltype(y))"
        end
        K = length(target_levels)
        μ = T.(log.(EvoTrees.proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == EvoTrees.GaussianMLE
        @assert eltype(y) <: Real
        K = 2
        y = T.(y)
        μ = [EvoTrees.mean(y), log(EvoTrees.std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == EvoTrees.LogisticMLE
        @assert eltype(y) <: Real
        K = 2
        y = T.(y)
        μ = [EvoTrees.mean(y), log(EvoTrees.std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        @assert eltype(y) <: Real
        K = 1
        y = T.(y)
        μ = [EvoTrees.mean(y)]
    end
    y = CuArray(y)
    μ = T.(μ)
    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)

    # initialize preds
    p = CUDA.zeros(T, K, nobs)
    p .= CuArray(μ)
    !isnothing(offset) && (pred .+= CuArray(offset'))

    # initialize gradients
    h∇_cpu = zeros(Float64, 2 * K + 1, maximum(featbins), length(featbins))
    h∇ = CuArray(h∇_cpu)
    ∇ = CUDA.zeros(T, 2 * K + 1, nobs)
    @assert (length(y) == length(w) && minimum(w) > 0)
    ∇[end, :] .= w

    # initialize indexes
    is_in = CUDA.zeros(UInt32, nobs)
    is_out = CUDA.zeros(UInt32, nobs)
    mask = CUDA.zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(eltype(js_), ceil(Int, config.colsample * nfeats))
    out = CUDA.zeros(UInt32, nobs)
    left = CUDA.zeros(UInt32, nobs)
    right = CUDA.zeros(UInt32, nobs)

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(params, :monotone_constraints) && for (k, v) in config.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    push!(params.info, :fnames => fnames)
    push!(params.info, :target_levels => target_levels)
    push!(params.info, :edges => edges)
    push!(params.info, :featbins => featbins)
    push!(params.info, :feattypes => feattypes)

    # initialize model
    nodes = [EvoTrees.TrainNode(featbins, K, view(is_in, 1:0)) for n = 1:2^config.max_depth-1]
    tree_bias = EvoTrees.Tree{L,K}(μ)
    push!(m.params.trees, tree_bias)

    # build cache
    push!(cache, :nrounds => 0)
    push!(cache, :x_bin => x_bin)
    push!(cache, :y => y)
    push!(cache, :w => w)
    push!(cache, :p => p)
    push!(cache, :K => K)
    push!(cache, :nodes => nodes)
    push!(cache, :is_in => is_in)
    push!(cache, :is_out => is_out)
    push!(cache, :mask => mask)
    push!(cache, :js_ => js_)
    push!(cache, :js => js)
    push!(cache, :out => out)
    push!(cache, :left => left)
    push!(cache, :right => right)
    push!(cache, :∇ => ∇)
    push!(cache, :h∇ => h∇)
    push!(cache, :h∇_cpu => h∇_cpu)
    push!(cache, :feattypes_gpu => CuArray(feattypes))
    push!(cache, :monotone_constraints => monotone_constraints)

    cache[:is_initialized] = true

    return nothing
end
