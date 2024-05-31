function LearnAPI.fit()
    @info "learnAPI fit!"
end

function LearnAPI.fit(algo::EvoTrees.EvoTypes{L}, data; kwargs...) where {L}
    @info "learnAPI fit"
    EvoTrees.fit_evotree(algo, data; kwargs...)
end
