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

m = EvoTreeRegressor(; loss=:mse, eta=1, max_nrounds=1)
EvoTrees.fit!(m, (x_train, y_train))
EvoTrees.fit!(m, dtrain; target_name)

# predict(m, x_train)
m(x_train)
