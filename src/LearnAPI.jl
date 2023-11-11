module LearnAPI

abstract type Model end

function fit!(m::Model)
    return nothing
end

function predict(m::Model, x)
    return nothing
end

function predict!(p, m::Model, x)
    return nothing
end

is_initialized(m::Model) = false

# Tables/MLJInterface-like approach to deterine model properties. Function to be extended by model implementation to specify support for feature.
# If is_iterative, then then fit!(m) method is implemented
is_iterative(m::Model) = false

# distinguish between iterative and online support. 
# - Iterative: data may have been subect to a preprocessing at init stage (binnings for histograms, or dataloaders creation for Flux models)
# - Online: new, unprocessed data can be provided to the model and be updated - involves the potential application of some data preprocessing
is_online(m::Model) = false

end #module