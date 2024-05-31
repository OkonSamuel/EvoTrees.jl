function LearnAPI.predict()
    @info "learnAPI pred!"
end

function LearnAPI.predict(m::EvoTree{L,K}, data) where {L,K}
    @info "learnAPI pred!"
    EvoTrees.predict(m, data, EvoTrees.CPU; ntree_limit=length(m.trees)) 
end
