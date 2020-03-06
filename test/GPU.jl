# using CUDA
using CUDAnative
using CuArrays
# using Flux
# using GeometricFlux

# CPU
function hist_cpu!(hist, δ, idx)
    Threads.@threads for j in 1:size(idx,2)
        @inbounds for i in 1:size(idx,1)
            hist[idx[i], j] += δ[i,j]
        end
    end
    return
end

# GPU - naive approach
function kernel!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    @inbounds if i <= size(id, 1) && j <= size(h, 2)
        k = Base._to_linear_index(h, id[i,j], j)
        CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

function hist_gpu!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}; MAX_THREADS=256) where {T<:AbstractFloat}
    thread_i = min(MAX_THREADS, size(id, 1))
    thread_j = min(MAX_THREADS ÷ thread_i, size(h, 2))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(h, 2)) ./ threads)
    CuArrays.@cuda blocks=blocks threads=threads kernel!(h, x, id)
    return h
end

nbins = 20
ncol = 100
items = Int(1e6)
hist = zeros(Float32, nbins, ncol)
δ = rand(Float32, items, ncol)
idx = rand(1:nbins, items, ncol)

hist_gpu = CuArray(hist)
δ_gpu = CuArray(δ)
idx_gpu = CuArray(idx)

@time hist_cpu!(hist, δ, idx)
@CuArrays.time hist_gpu!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=512)




# GPU - apply along the features axis
function kernel_y!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = threadIdx().y + (blockIdx().y - 1) * blockDim().y
    @inbounds if i <= size(id, 1) && j <= size(h, 2)
        k = Base._to_linear_index(h, id[i,j], j)
        CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

function hist_gpu_y!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}; MAX_THREADS=256) where {T<:AbstractFloat}
    thread_j = min(MAX_THREADS, size(id, 2))
    thread_i = min(MAX_THREADS ÷ thread_j, size(h, 1))
    threads = (thread_i, thread_j)
    blocks = ceil.(Int, (size(id, 1), size(h, 2)) ./ threads)
    CuArrays.@cuda blocks=blocks threads=threads kernel!(h, x, id)
    return h
end

# blockdim: (THREADS,1)
@CuArrays.time hist_gpu_y!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=1024)

function test(h)
    t = zero(eltype(h))
    for i in eachindex(h)
        t += i
    end
    return t
end
test(hist_gpu)



# GPU - 2-step approach
function kernel_1!(h::CuDeviceMatrix{T}, x::CuDeviceMatrix{T}, id) where {T<:AbstractFloat}
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    j = blockIdx().y
    @inbounds if i <= size(id, 1) && j <= size(h, 2)
        k = Base._to_linear_index(h, id[i,j], j)
        CUDAnative.atomic_add!(pointer(h, k), x[i,j])
    end
    return
end

function hist_gpu!(h::CuMatrix{T}, x::CuMatrix{T}, id::CuMatrix{Int}; MAX_THREADS=256) where {T<:AbstractFloat}
    threads = min(MAX_THREADS, size(id, 1))
    blocks = ceil(Int, size(id, 1) ./ threads), size(h, 2)
    println("threads: ", threads)
    println("blocks: ", blocks)
    CuArrays.@cuda blocks=blocks threads=threads kernel_1!(h, x, id)
    return h
end

# blockdim: (THREADS,1)
@CuArrays.time hist_gpu!(hist_gpu, δ_gpu, idx_gpu, MAX_THREADS=256)
