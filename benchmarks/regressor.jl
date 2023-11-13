using Revise
using Statistics
using StatsBase: sample
using XGBoost
# using LightGBM
using EvoTrees
using BenchmarkTools
using Random: seed!
# import CUDA

### v.0.15.1
# desktop | 1e6 | depth 11 | cpu: 37.2s
# desktop | 10e6 | depth 11 | cpu

### v0.16.5
# desktop | 1e6 | depth 11 | cpu: 31s gpu: 50 sec  | xgboost cpu: 26s
# desktop | 10e6 | depth 11 | cpu 200s gpu: 80 sec | xgboost cpu: 267s

#threads
# laptop depth 6: 12.717845 seconds (2.08 M allocations: 466.228 MiB)

nobs = Int(1e6)
num_feat = Int(100)
max_nrounds = 200
max_depth = 6
tree_type = "binary"
T = Float64
nthread = Base.Threads.nthreads()
@info "testing with: $nobs observations | $num_feat features. nthread: $nthread | tree_type : $tree_type | max_depth : $max_depth"
seed!(123)
x_train = rand(T, nobs, num_feat)
y_train = rand(T, size(x_train, 1))

@info nthread
loss = "mse"
if loss == "mse"
    loss_xgb = "reg:squarederror"
    metric_xgb = "mae"
    loss_evo = :mse
    metric_evo = :mae
elseif loss == "logloss"
    loss_xgb = "reg:logistic"
    metric_xgb = "logloss"
    loss_evo = :logloss
    metric_evo = :logloss
end

@info "XGBoost"
@info "train"
params_xgb = Dict(
    :num_round => max_nrounds,
    :max_depth => max_depth - 1,
    :eta => 0.05,
    :objective => loss_xgb,
    :print_every_n => 5,
    :subsample => 0.5,
    :colsample_bytree => 0.5,
    :tree_method => "hist", # hist/gpu_hist
    :max_bin => 64,
)

dtrain = DMatrix(x_train, y_train)
watchlist = Dict("train" => DMatrix(x_train, y_train));
@time m_xgb = xgboost(dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric=metric_xgb, params_xgb...);
# @btime m_xgb = xgboost($dtrain; watchlist, nthread=nthread, verbosity=0, eval_metric = metric_xgb, params_xgb...);
@info "predict"
@time pred_xgb = XGBoost.predict(m_xgb, x_train);
# @btime XGBoost.predict($m_xgb, $x_train);

# @info "lightgbm train:"
# m_gbm = LGBMRegression(
#     objective = "regression",
#     boosting = "gbdt",
#     num_iterations = 200,
#     learning_rate = 0.05,
#     num_leaves = 256,
#     max_depth = 5,
#     tree_learner = "serial",
#     num_threads = Sys.CPU_THREADS,
#     histogram_pool_size = -1.,
#     min_data_in_leaf = 1,
#     min_sum_hessian_in_leaf = 0,
#     max_delta_step = 0,
#     min_gain_to_split = 0,
#     feature_fraction = 0.5,
#     feature_fraction_seed = 2,
#     bagging_fraction = 0.5,
#     bagging_freq = 1,
#     bagging_seed = 3,
#     max_bin = 64,
#     bin_construct_sample_cnt = 200000,
#     data_random_seed = 1,
#     is_sparse = false,
#     feature_pre_filter = false,
#     is_unbalance = false,
#     min_data_per_group = 1,
#     metric = ["mae"],
#     metric_freq = 10,
#     # early_stopping_round = 10,
# )
# @time gbm_results = fit!(m_gbm, x_train, y_train, (x_train, y_train))
# @time pred_gbm = LightGBM.predict(m_gbm, x_train) |> vec

@info "EvoTrees"
verbosity = 1

evo_kw = Dict(
    :loss => loss_evo,
    :max_nrounds => max_nrounds,
    :max_depth => max_depth,
    :tree_type => tree_type,
    :lambda => 0.0,
    :gamma => 0.0,
    :eta => 0.05,
    :min_weight => 1.0,
    :rowsample => 0.5,
    :colsample => 0.5,
    :max_bins => 64,
    :rng => 123
)

@info "EvoTrees CPU"
device = "cpu"

# @info "init"
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.init!(m_evo, (x_train, y_train); device);
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.init!(m_evo, (x_train, y_train); device);

# @info "train - no eval"
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.fit!(m_evo, (x_train, y_train); device);
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.fit!(m_evo, (x_train, y_train); device);

@info "train - eval"
m_evo = EvoTreeRegressor(; evo_kw...)
@time EvoTrees.fit!(m_evo, (x_train, y_train), (x_train, y_train); metric=metric_evo, device, verbosity, print_every_n=100);
m_evo = EvoTreeRegressor(; evo_kw...)
@time EvoTrees.fit!(m_evo, (x_train, y_train), (x_train, y_train); metric=metric_evo, device, verbosity, print_every_n=100);

@info "predict"
@time pred_evo = m_evo(x_train);
@time pred_evo = m_evo(x_train);


@info "EvoTrees GPU"
device = "gpu"

# @info "init"
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.init!(m_evo, (x_train, y_train); device, verbosity);
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.init!(m_evo, (x_train, y_train); device, verbosity);

# @info "train - no eval"
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.fit!(m_evo, (x_train, y_train); device, verbosity);
# m_evo = EvoTreeRegressor(; evo_kw...)
# @time EvoTrees.fit!(m_evo, (x_train, y_train); device, verbosity);

@info "train - eval"
m_evo = EvoTreeRegressor(; evo_kw...)
@time EvoTrees.fit!(m_evo, (x_train, y_train), (x_train, y_train); metric=metric_evo, device, verbosity, print_every_n=100);
m_evo = EvoTreeRegressor(; evo_kw...)
@time EvoTrees.fit!(m_evo, (x_train, y_train), (x_train, y_train); metric=metric_evo, device, verbosity, print_every_n=100);

@info "predict"
@time pred_evo = m_evo(x_train; device);
@time pred_evo = m_evo(x_train; device);
