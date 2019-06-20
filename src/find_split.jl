#############################################
# Get the braking points
#############################################
function get_edges(X, nbins=250)
    edges = Vector{Vector}(undef, size(X,2))
    @threads for i in 1:size(X, 2)
        edges[i] = unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:(end-1)]
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :,i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i in 1:size(X, 2)
        X_bin[:,i] = searchsortedlast.(Ref(edges[i]), view(X,:,i)) .+ 1
    end
    X_bin
end

function find_bags(x::Vector{T}, edges::Vector{T}) where T<:Real
    idx = BitSet(1:length(x) |> collect)
    bags = [BitSet() for _ in 1:length(edges)]
    for i in idx
        bin = 1
        while x[i] > edges[bin]
            bin +=1
        end
        union!(bags[bin], i)
    end
    return bags
end

function update_bags!(bins, set)
    for bin in bins
        intersect!(bin, set)
    end
end

function find_split_bitset!(bins, δ::Vector{S}, δ²::Vector{S}, 𝑤::Vector{S}, ∑δ::S, ∑δ²::S, ∑𝑤::S, λ::S, info::SplitInfo{S, Int}, track::SplitTrack{S}, edges, set::BitSet) where {S<:AbstractFloat}

    info.gain = get_gain(∑δ, ∑δ², ∑𝑤, λ)

    track.∑δL = zero(S)
    track.∑δ²L = zero(S)
    track.∑𝑤L = zero(S)
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    @inbounds for bin in 1:(length(bins)-1)
        @inbounds for i in bins[bin]
            if i in set
                track.∑δL += δ[i]
                track.∑δ²L += δ²[i]
                track.∑𝑤L += 𝑤[i]
                track.∑δR -= δ[i]
                track.∑δ²R -= δ²[i]
                track.∑𝑤R -= 𝑤[i]
            end
        end
        update_track!(track, λ)
        # if gain > info.gain && ∑𝑤R > zero(S)
        if track.gain > info.gain && track.∑𝑤R > zero(S)
        # if track.gain > info.gain
            info.gain = track.gain
            info.gainL = track.gainL
            info.gainR = track.gainR
            info.∑δL = track.∑δL
            info.∑δ²L = track.∑δ²L
            info.∑𝑤L = track.∑𝑤L
            info.∑δR = track.∑δR
            info.∑δ²R = track.∑δ²R
            info.∑𝑤R = track.∑𝑤R
            info.cond = edges[bin]
            info.𝑖 = bin
        end
    end
    return
end

# find best split on binarized data
function find_split_bin!(x::AbstractArray{T, 1}, δ::AbstractArray{Float64, 1}, δ²::AbstractArray{Float64, 1}, 𝑤::AbstractArray{Float64, 1}, ∑δ, ∑δ², ∑𝑤, params::EvoTreeRegressor, info::SplitInfo, track::SplitTrack, x_edges) where T<:Real

    info.gain = get_gain(params.loss, ∑δ, ∑δ², ∑𝑤, params.λ)

    track.∑δL = 0.0
    track.∑δ²L = 0.0
    track.∑𝑤L = 0.0
    track.∑δR = ∑δ
    track.∑δ²R = ∑δ²
    track.∑𝑤R = ∑𝑤

    @inbounds for i in 1:(size(x, 1) - 1)
        track.∑δL += δ[i]
        track.∑δ²L += δ²[i]
        track.∑𝑤L += 𝑤[i]
        track.∑δR -= δ[i]
        track.∑δ²R -= δ²[i]
        track.∑𝑤R -= 𝑤[i]

        @inbounds if x[i] < x[i+1] && track.∑𝑤L >= params.min_weight && track.∑𝑤R >= params.min_weight # check gain only if there's a change in value
            update_track!(params.loss, track, params.λ)
            if track.gain > info.gain
                info.gain = track.gain
                info.gainL = track.gainL
                info.gainR = track.gainR
                info.∑δL = track.∑δL
                info.∑δ²L = track.∑δ²L
                info.∑𝑤L = track.∑𝑤L
                info.∑δR = track.∑δR
                info.∑δ²R = track.∑δ²R
                info.∑𝑤R = track.∑𝑤R
                info.cond = x_edges[x[i]]
                info.𝑖 = i
            end
        end
    end
end
