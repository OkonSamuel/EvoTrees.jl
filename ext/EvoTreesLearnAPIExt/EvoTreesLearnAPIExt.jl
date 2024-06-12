module EvoTreesLearnAPIExt

using EvoTrees

using LearnAPI
import LearnAPI: fit, predict

function LearnAPI.fit()
    @info "EvoTreesLearnAPIExt fit"
end
function LearnAPI.fit(algo::EvoTrees.EvoTypes{L}, data; kwargs...) where {L}
    @info "EvoTreesLearnAPIExt fit"
    EvoTrees.fit_evotree(algo, data; kwargs...)
end

function LearnAPI.predict()
    @info "EvoTreesLearnAPIExt predict"
end
function LearnAPI.predict(m::EvoTree{L,K}, data) where {L,K}
    @info "EvoTreesLearnAPIExt predict"
    EvoTrees.predict(m, data, EvoTrees.CPU; ntree_limit=length(m.trees)) 
end

end # module
