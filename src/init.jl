function init_core!(m::EvoTree, ::Type{CPU}, data, fnames, y, w, offset)

    config = m.config
    params = m.params
    cache = m.cache
    L = config.loss_type

    # binarize data into quantiles
    edges, featbins, feattypes = get_edges(data; fnames, max_bins=config.max_bins, rng=config.rng)
    x_bin = binarize(data; fnames, edges)
    nobs, nfeats = size(x_bin)
    T = Float32

    target_levels = nothing
    if L == Logistic
        @assert eltype(y) <: Real && minimum(y) >= 0 && maximum(y) <= 1
        K = 1
        y = T.(y)
        μ = [logit(mean(y))]
        !isnothing(offset) && (offset .= logit.(offset))
    elseif L in [Poisson, Gamma, Tweedie]
        @assert eltype(y) <: Real
        K = 1
        y = T.(y)
        μ = fill(log(mean(y)), 1)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == MLogLoss
        if eltype(y) <: CategoricalValue
            target_levels = CategoricalArrays.levels(y)
            y = UInt32.(CategoricalArrays.levelcode.(y))
        elseif eltype(y) <: Integer || eltype(y) <: Bool || eltype(y) <: String || eltype(y) <: Char
            target_levels = sort(unique(y))
            yc = CategoricalVector(y, levels=target_levels)
            y = UInt32.(CategoricalArrays.levelcode.(yc))
        else
            @error "Invalid target eltype: $(eltype(y))"
        end
        K = length(target_levels)
        μ = T.(log.(proportions(y, UInt32(1):UInt32(K))))
        μ .-= maximum(μ)
        !isnothing(offset) && (offset .= log.(offset))
    elseif L == GaussianMLE
        @assert eltype(y) <: Real
        K = 2
        y = T.(y)
        μ = [mean(y), log(std(y))]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    elseif L == LogisticMLE
        @assert eltype(y) <: Real
        K = 2
        y = T.(y)
        μ = [mean(y), log(std(y) * sqrt(3) / π)]
        !isnothing(offset) && (offset[:, 2] .= log.(offset[:, 2]))
    else
        @assert eltype(y) <: Real
        K = 1
        y = T.(y)
        μ = [mean(y)]
    end
    μ = T.(μ)

    # force a neutral/zero bias/initial tree when offset is specified
    !isnothing(offset) && (μ .= 0)
    @assert (length(y) == length(w) && minimum(w) > 0)

    # initialize preds
    p = zeros(T, K, nobs)
    p .= μ
    !isnothing(offset) && (p .+= offset')

    # initialize gradients
    ∇ = zeros(T, 2 * K + 1, nobs)
    ∇[end, :] .= w

    # initialize indexes
    is_in = zeros(UInt32, nobs)
    is_out = zeros(UInt32, nobs)
    mask = zeros(UInt8, nobs)
    js_ = UInt32.(collect(1:nfeats))
    js = zeros(UInt32, ceil(Int, config.colsample * nfeats))
    out = zeros(UInt32, nobs)
    left = zeros(UInt32, nobs)
    right = zeros(UInt32, nobs)

    # assign monotone contraints in constraints vector
    monotone_constraints = zeros(Int32, nfeats)
    hasproperty(config, :monotone_constraints) && for (k, v) in config.monotone_constraints
        monotone_constraints[k] = v
    end

    # model info
    push!(params.info, :fnames => fnames)
    push!(params.info, :target_levels => target_levels)
    push!(params.info, :edges => edges)
    push!(params.info, :featbins => featbins)
    push!(params.info, :feattypes => feattypes)

    # initialize model
    nodes = [TrainNode(featbins, K, view(is_in, 1:0)) for n = 1:2^config.max_depth-1]
    tree_bias = Tree{L,K}(μ)
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
    push!(cache, :monotone_constraints => monotone_constraints)

    cache[:is_initialized] = true

    return nothing
end

"""
    init(
        m::EvoTree,
        dtrain,
        device::Type{<:Device}=CPU;
        target_name,
        fnames=nothing,
        w_name=nothing,
        offset_name=nothing
    )

Initialise EvoTree
"""
function init!(
    m::EvoTree,
    dtrain;
    device::Type{<:Device}=CPU,
    target_name,
    fnames=nothing,
    w_name=nothing,
    offset_name=nothing,
    kwargs...
)

    # set fnames
    schema = Tables.schema(dtrain)
    _w_name = isnothing(w_name) ? Symbol("") : Symbol(w_name)
    _offset_name = isnothing(offset_name) ? Symbol("") : Symbol(offset_name)
    _target_name = Symbol(target_name)
    if isnothing(fnames)
        fnames = Symbol[]
        for i in eachindex(schema.names)
            if schema.types[i] <: Union{Real,CategoricalValue}
                push!(fnames, schema.names[i])
            end
        end
        fnames = setdiff(fnames, union([_target_name], [_w_name], [_offset_name]))
    else
        isa(fnames, String) ? fnames = [fnames] : nothing
        fnames = Symbol.(fnames)
        @assert isa(fnames, Vector{Symbol})
        @assert all(fnames .∈ Ref(schema.names))
        for name in fnames
            @assert schema.types[findfirst(name .== schema.names)] <: Union{Real,CategoricalValue}
        end
    end

    T = Float32
    nobs = length(Tables.getcolumn(dtrain, 1))
    y_train = Tables.getcolumn(dtrain, _target_name)
    V = device_array_type(device)
    w = isnothing(w_name) ? device_ones(device, T, nobs) : V{T}(Tables.getcolumn(dtrain, _w_name))
    offset = isnothing(offset_name) ? nothing : V{T}(Tables.getcolumn(dtrain, _offset_name))

    init_core!(m, device, dtrain, fnames, y_train, w, offset)

    return nothing
end

# This should be different on CPUs and GPUs
device_ones(::Type{<:CPU}, ::Type{T}, n::Int) where {T} = ones(T, n)
device_array_type(::Type{<:CPU}) = Array

"""
    init(
        params::EvoTypes,
        dtrain::Tuple{Matrix,Vector};
        device::Type{<:Device}=CPU;
        fnames=nothing,
        w_train=nothing,
        offset_train=nothing
    )

Initialise EvoTree
"""
function init!(
    m::EvoTree,
    dtrain::Tuple{Matrix,Vector};
    device::Type{<:Device}=CPU,
    fnames=nothing,
    w_train=nothing,
    offset_train=nothing,
    kwargs...
)

    x_train, y_train = dtrain
    fnames = isnothing(fnames) ? [Symbol("feat_$i") for i in axes(x_train, 2)] : Symbol.(fnames)
    @assert length(fnames) == size(x_train, 2)

    T = Float32
    nobs = size(x_train, 1)
    V = device_array_type(device)
    w = isnothing(w_train) ? device_ones(device, T, nobs) : V{T}(w_train)
    offset = isnothing(offset_train) ? nothing : V{T}(offset_train)

    init_core!(m, device, x_train, fnames, y_train, w, offset)

    return nothing
end
