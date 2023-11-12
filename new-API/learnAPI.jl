using Revise
using EvoTrees
using DataFrames
# using EvoTrees.LearnAPI
# using GLM

nobs = 1_000
nfeats = 10
x_train = rand(nobs, nfeats);
y_train = rand(nobs);

dtrain = DataFrame(x_train, :auto);
dtrain.y = y_train;
target_name = "y"

m = EvoTreeRegressor(; loss=:mse)
EvoTrees.fit!(m, (x_train, y_train))
m(x_train)

m = EvoTreeRegressor(; loss=:mse)
EvoTrees.fit!(m, dtrain; target_name)
m(dtrain)

device="gpu"
m = EvoTreeRegressor(; loss=:mse)
EvoTrees.fit!(m, (x_train, y_train); device)
EvoTrees.fit!(m, dtrain; device, target_name)
@time m(x_train; device)
@time m(x_train; device)
