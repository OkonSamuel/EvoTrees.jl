# # Exploring a credibility-based approach for tree-gain estimation

#=
The motivation for this experiment stems from potential shortfalls in approach used in gradoent-boosted trees to assess the best split potential. 
=#

#=
The idea is for the *gain* to capture the varying level of uncertainty associated with the observations associated within a tree-split candidate.
=#

## compare mse vs cred based metrics
using EvoTrees
using EvoTrees: get_gain, _get_cred
using DataFrames
using Distributions
using Statistics: mean, std
using CairoMakie

function get_∑(p::Matrix{T}, y::Vector{T}, params) where {T}
    ∇ = Matrix{T}(undef, 3, length(y))
    view(∇, 3, :) .= 1
    EvoTrees.update_grads!(∇, p, y, params)
    ∑ = dropdims(sum(∇, dims=2), dims=2)
    return ∑
end

function simul_Z(; nobs, loss, spread=1.0, sd=1.0)
    config = EvoTreeRegressor(; loss)
    p = zeros(1, nobs)
    y = randn(nobs)
    _std = length(y) == 1 ? abs(first(y)) : std(y; corrected=false)
    y .= (y .- mean(y)) ./ _std .* sd .- spread
    ∑ = get_∑(p, y, config)
    Z = EvoTrees._get_cred(config, ∑)
    return Z
end


function get_data(; loss, nobs, spread=1.0, sd=1.0, lambda=0.0)
    yL, yR = randn(nobs), randn(nobs)
    yL .= (yL .- mean(yL)) ./ std(yL) .* sd .- spread / 2
    yR .= (yR .- mean(yR)) ./ std(yR) .* sd .+ spread / 2
    yT = vcat(yL, yR)

    pL = zeros(1, nobs)
    pR = zeros(1, nobs)
    pT = zeros(1, nobs)

    data = Dict()
    data[:yL] = yL
    data[:yR] = yR

    ## Creds
    config = EvoTreeRegressor(; loss)
    ∑T = get_∑(pT, yT, config)
    ∑L = get_∑(pL, yL, config)
    ∑R = get_∑(pR, yR, config)
    data[:gP] = get_gain(config, ∑T)
    data[:gL] = get_gain(config, ∑L)
    data[:gR] = get_gain(config, ∑R)
    data[:gC] = data[:gL] + data[:gR]

    if loss != :mse
        data[:ZR] = _get_cred(config, ∑R)
    else
        data[:ZR] = NaN
    end

    return data
end

function get_dist_figure(data)

    gP = round(data[:gP]; digits=3)
    gC = round(data[:gC]; sigdigits=4)
    gL = round(data[:gL]; sigdigits=4)
    gR = round(data[:gR]; sigdigits=4)
    ZR = round(data[:ZR]; sigdigits=4)

    f = Figure()
    ax1 = Axis(f[1, 1];
        subtitle=
        """
        gain parent=$gP | gain cildren=$gC\n
        gainL=$gL | gainR=$gR | ZR=$ZR
        """
    )
    density!(ax1, data[:yL]; color="#02723599", label="left")
    density!(ax1, data[:yR]; color="#02357299", label="right")
    Legend(f[2, 1], ax1, orientation=:horizontal)
    return f
end

function get_cred_figure(;
    loss,
    sd,
    nobs_list,
    spread_list)

    xticks = string.(nobs_list)
    yticks = string.(spread_list)

    matrix = zeros(length(nobs_list), length(spread_list))

    for (idx, nobs) in enumerate(nobs_list)
        for (idy, spread) in enumerate(spread_list)
            @info "nobs: $nobs | spread: $spread"
            z = simul_Z(; loss, nobs, spread, sd)
            matrix[idx, idy] = z
        end
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; title="$(string(loss)) | sd: $sd", xlabel="nobs", ylabel="spread", xticks=(1:length(xticks), xticks), yticks=(1:length(yticks), yticks))
    heat = heatmap!(ax, matrix)
    Colorbar(fig[2, 1], heat; vertical=false)
    return fig
end

#=
## Credibility-based gains
=#
loss = :mse
data = get_data(; loss, nobs=10, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-mse-1.png"), f)

data = get_data(; loss, nobs=100, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-mse-2.png"), f)

data = get_data(; loss, nobs=10000, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-mse-3.png"), f)

#=
| ![](assets/dist-mse-1.png) | ![](assets/dist-mse-2.png) | ![](assets/dist-mse-3.png) |
|:----------------------:|:----------------------:|:----------------------:|
=#

#=
## Credibility-based gains
=#
loss = :credV1A
data = get_data(; loss, nobs=10, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-cred-1.png"), f)

data = get_data(; loss, nobs=100, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-cred-2.png"), f)

data = get_data(; loss, nobs=10000, spread=1.0, sd=1.0, lambda=0.0)
f = get_dist_figure(data)
save(joinpath(@__DIR__, "assets", "dist-cred-3.png"), f)

#=
| ![](assets/dist-cred-1.png) | ![](assets/dist-cred-2.png) | ![](assets/dist-cred-3.png) |
|:----------------------:|:----------------------:|:----------------------:|
=#

#=
credibility figures
=#
sd = 1.0
nobs_list = Int.(10.0 .^ (0:6))
nobs_list[1] = 2
spread_list = [0.001, 0.01, 0.1, 0.5, 1, 2, 10, 100]

get_cred_figure(; loss=:credV1A, sd, nobs_list, spread_list)
get_cred_figure(; loss=:credV2A, sd, nobs_list, spread_list)

get_cred_figure(; loss=:credV1B, sd, nobs_list, spread_list)
get_cred_figure(; loss=:credV2B, sd, nobs_list, spread_list)

f1 = get_cred_figure(; loss=:credV1A, sd, nobs_list, spread_list)
f2 = get_cred_figure(; loss=:credV2A, sd, nobs_list, spread_list)
save(joinpath(@__DIR__, "test.png"), f1)
f1