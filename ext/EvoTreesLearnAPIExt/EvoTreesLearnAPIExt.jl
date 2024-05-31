module EvoTreesLearnAPIExt

using EvoTrees

using LearnAPI
import LearnAPI: fit, predict

include("predict.jl")
include("fit.jl")

end # module
