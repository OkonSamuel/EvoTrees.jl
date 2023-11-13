abstract type LossType end
abstract type ModelType <: LossType end
abstract type GradientRegression <: LossType end
abstract type MLE2P <: LossType end # 2-parameters max-likelihood

abstract type MSE <: GradientRegression end
abstract type LogLoss <: GradientRegression end
abstract type Poisson <: GradientRegression end
abstract type Gamma <: GradientRegression end
abstract type Tweedie <: GradientRegression end
abstract type MLogLoss <: LossType end
abstract type GaussianMLE <: MLE2P end
abstract type LogisticMLE <: MLE2P end
abstract type Quantile <: LossType end
abstract type L1 <: LossType end

# Converts MSE -> :mse
const _type2loss_dict = Dict(
    MSE => :mse,
    LogLoss => :logloss,
    Poisson => :poisson,
    Gamma => :gamma,
    Tweedie => :tweedie,
    MLogLoss => :mlogloss,
    GaussianMLE => :gaussian_mle,
    LogisticMLE => :logistic_mle,
    Quantile => :quantile,
    L1 => :l1,
)
_type2loss(L::Type) = _type2loss_dict[L]

const _loss2type_dict = Dict(
    :mse => MSE,
    :logloss => LogLoss,
    :gamma => Gamma,
    :tweedie => Tweedie,
    :poisson => Poisson,
    :mlogloss => MLogLoss,
    :gaussian_mle => GaussianMLE,
    :logistic_mle => LogisticMLE,
    :quantile => Quantile,
    :l1 => L1
)
_loss2type(loss::Symbol) = _loss2type_dict[loss]

# make a Random Number Generator object
mk_rng(rng::AbstractRNG) = rng
mk_rng(int::Integer) = Random.MersenneTwister(int)

struct Config
    loss_type::Type{<:LossType}
    outsize::Int
    loss::Symbol
    max_nrounds::Int
    L2::Float64
    lambda::Float64
    gamma::Float64
    eta::Float64
    max_depth::Int
    min_weight::Float64 # real minimum number of observations, different from xgboost (but same for linear)
    rowsample::Float64 # subsample
    colsample::Float64
    max_bins::Int
    alpha::Float64
    monotone_constraints::Any
    tree_type::String
    rng::Any
end


function Config(; kwargs...)
    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :max_nrounds => 100,
        :L2 => 0.0,
        :lambda => 0.0,
        :gamma => 0.0, # min gain to split
        :eta => 0.1, # learning rate
        :max_depth => 6,
        :min_weight => 1.0, # minimal weight, different from xgboost (but same for linear)
        :rowsample => 1.0,
        :colsample => 1.0,
        :max_bins => 64,
        :alpha => 0.5,
        :monotone_constraints => Dict{Int,Int}(),
        :tree_type => "binary",
        :rng => 123,
    )
    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    args[:loss] = Symbol(args[:loss])

    if args[:loss] ∉ keys(_loss2type_dict)
        error("Invalid loss: $(args[:loss]). Must be one of $(keys(_loss2type_dict)).")
    end
    L = _loss2type(args[:loss])
    K = L <: MLE2P ? 2 : 1

    config = Config(
        L,
        K,
        args[:loss],
        args[:max_nrounds],
        args[:L2],
        args[:lambda],
        args[:gamma],
        args[:eta],
        args[:max_depth],
        args[:min_weight],
        args[:rowsample],
        args[:colsample],
        args[:max_bins],
        args[:alpha],
        args[:monotone_constraints],
        args[:tree_type],
        args[:rng],
    )

    check_args(config)
    return config
end

function Base.show(io::IO, config::Config)
    println(io, "$(typeof(config))")
    for fname in fieldnames(typeof(config))
        println(io, " - $fname: $(getfield(config, fname))")
    end
end

# <:MMI.ModelParams
struct Params
    loss_type::Type{<:LossType}
    outsize::Int
    info::Dict{Symbol,Any}
    trees::Vector{<:Tree}
end

function Params(config::Config)
    L = config.loss_type
    K = config.outsize
    info = Dict{Symbol,Any}()
    trees = Tree{L,K}[]
    params = Params(L, K, info, trees)
    return params
end

struct EvoTreeRegressor <: MMI.Deterministic
    config::Config
    params::Params
    cache::Dict{Symbol,Any}
end
function EvoTreeRegressor(; kw...)
    config = Config(; kw...)
    params = Params(config)
    cache = Dict{Symbol,Any}(:is_initialized => false)
    return EvoTreeRegressor(config, params, cache)
end

struct EvoTreeCount <: MMI.Deterministic
    config::Config
    params::Params
    cache::Dict{Symbol,Any}
end
function EvoTreeCount(; kw...)
    config = Config(; kw...)
    params = Params(config)
    cache = Dict{Symbol,Any}(:is_initialized => false)
    return EvoTreeCount(config, params, cache)
end

struct EvoTreeClassifier <: MMI.Deterministic
    config::Config
    params::Params
    cache::Dict{Symbol,Any}
end
function EvoTreeClassifier(; kw...)
    config = Config(; kw...)
    params = Params(config)
    cache = Dict{Symbol,Any}(:is_initialized => false)
    return EvoTreeClassifier(config, params, cache)
end

struct EvoTreeMLE <: MMI.Deterministic
    config::Config
    params::Params
    cache::Dict{Symbol,Any}
end
function EvoTreeMLE(; kw...)
    config = Config(; kw...)
    params = Params(config)
    cache = Dict{Symbol,Any}(:is_initialized => false)
    return EvoTreeMLE(config, params, cache)
end

const EvoTree = Union{
    EvoTreeRegressor,
    EvoTreeCount,
    EvoTreeClassifier,
    EvoTreeMLE
}

is_initialized(m::EvoTree) = m.cache[:is_initialized]

function Base.show(io::IO, m::EvoTree)
    println(io, "# $(typeof(m))")
    println(io, m.config)
    println(io, " - Contains $(length(m.params.trees)) trees in field `params.trees` (incl. 1 bias tree).")
    if is_initialized(m)
        println(io, " - Data input has $(length(m.params.info[:fnames])) features.")
        println(io, " - $(keys(evotree.params.info)) info accessible in field `params.info`")
    end
end

"""
    Inference / prediction overloading for EvoTree models
"""
function (m::EvoTree)(data; ntree_limit=length(m.params.trees), device="cpu")
    _device = string(device) == "gpu" ? GPU : CPU
    return _predict(m, data, _device; ntree_limit)
end


"""
    check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}
Check model parameter if it's valid
"""
function check_parameter(::Type{<:T}, value, min_value::Real, max_value::Real, label::Symbol) where {T<:Number}
    min_value = max(typemin(T), min_value)
    max_value = min(typemax(T), max_value)
    try
        convert(T, value)
        @assert min_value <= value <= max_value
    catch
        error("Invalid value for parameter `$(string(label))`: $value. `$(string(label))` must be of type $T with value between $min_value and $max_value.")
    end
end


"""
    check_args(config::Config)

Check model arguments if they are valid (eg, after mutation when tuning hyperparams)
Note: does not check consistency of model type and loss selected
"""
function check_args(config::Config)

    check_parameter(Int, config.max_depth, 1, typemax(Int), :max_depth)
    check_parameter(Int, config.max_nrounds, 0, typemax(Int), :max_nrounds)
    check_parameter(Int, config.max_bins, 2, 255, :max_bins)

    check_parameter(Float64, config.lambda, zero(Float64), typemax(Float64), :lambda)
    check_parameter(Float64, config.gamma, zero(Float64), typemax(Float64), :gamma)
    check_parameter(Float64, config.min_weight, zero(Float64), typemax(Float64), :min_weight)

    check_parameter(Float64, config.alpha, zero(Float64), one(Float64), :alpha)
    check_parameter(Float64, config.rowsample, eps(Float64), one(Float64), :rowsample)
    check_parameter(Float64, config.colsample, eps(Float64), one(Float64), :colsample)
    check_parameter(Float64, config.eta, zero(Float64), typemax(Float64), :eta)

    try
        tree_type = string(config.tree_type)
        @assert tree_type ∈ ["binary", "oblivious"]
    catch
        error("Invalid input for `tree_type` parameter: `$(config.tree_type)`. Must be of one of `binary` or `oblivious`")
    end
end