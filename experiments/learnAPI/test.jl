using LearnAPI
using EvoTrees

using MLDatasets
using DataFrames
using Statistics: mean
using CategoricalArrays
using Random

using LearnAPI: constructor
# check that EvoTreesLearnAPIExt is properly loaded and learnAP's fit/predict methods are overloaded
fit()
predict()

# test constructor - need to fix the parametric type: changes from LogLoss to MSE
algorithm = EvoTreeRegressor(loss=:logistic)
constructor(algorithm)(lambda=0.05)

df = MLDatasets.BostonHousing().dataframe
Random.seed!(123)

train_ratio = 0.8
train_indices = randperm(nrow(df))[1:Int(round(train_ratio * nrow(df)))]

dtrain = df[train_indices, :]
deval = df[setdiff(1:nrow(df), train_indices), :]

target_name = "MEDV"
fnames = setdiff(names(df), [target_name])

config = EvoTreeRegressor(
    nrounds=200, 
    eta=0.1, 
    max_depth=4, 
    lambda=0.1, 
    rowsample=0.9, 
    colsample=0.9)

####################################################################
# use learnAPI basic fit / predict
####################################################################
model = fit(config, dtrain; target_name, fnames)

# use LearnAPI predict
pred_train = predict(model, dtrain)
pred_eval = predict(model, deval)

# 1.056997874224627
mean(abs.(pred_train .- dtrain[!, target_name]))
# 2.3298767665825264
mean(abs.(pred_eval .- deval[!, target_name]))

####################################################################
# use learnAPI fit with deval / early stopping
####################################################################
model = fit(config, dtrain; target_name, fnames, deval, device=:cpu, metric="mse", early_stopping_rounds=10, print_every_n=25)

# use LearnAPI predict
pred_train = predict(model, dtrain)
pred_eval = predict(model, deval)

mean(abs.(pred_train .- dtrain[!, target_name]))
mean(abs.(pred_eval .- deval[!, target_name]))
