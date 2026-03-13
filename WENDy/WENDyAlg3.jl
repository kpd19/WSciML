#WendyAlg3

using LinearAlgebra
using FFTW # fast fourier transform
using SparseArrays
using BlockDiagonals
using Symbolics
using DifferentialEquations
using DSP          # convolve
using Statistics   # mean
using SpecialFunctions  # gamma, besselj
using Combinatorics     # binomial (comb)
using Polynomials
using HypergeometricFunctions
using ModelingToolkit
using Roots
using Interpolations

function print_diagnostics()

    err_norm = 2
    errs = [norm(w_hat_its[:, i] .- true_params, err_norm) / norm(true_params, err_norm)
            for i in 1:size(w_hat_its, 2)]
    err_wendy = last(errs)
    println("err WENDy: "*string(err_wendy))
    println("parameters: true, WENDy")
    hcat(w_hat, true_params)
    G = RT \ G_0
    b = RT\ b_0
    res_true = G*true_params-b_0
    err_NLS = 0
    total_time_nls = 0
    param_nls = nothing

    println("-----------------")
    println("")
    println("time elapsed: "*string(total_time)*" seconds")
    println("# iterations: "*string(size(w_hat_its)[2]))
    println("K, d, M: ", string(size(V_cell[1])[1]), ", ", string(nstates), ", ", string(length(tobs)))
    println("test function radii: "*string(length(unique(vec(mt)))))
    println("")
    println("-----------------")

    x_cell = [xsub[:, i] for i in 1:nstates]

    Theta_cell_true = [hcat([[f(xsub[i, :]) for i in 1:M] for f in flist]...)
                    for flist in features]

    G_0_true = Matrix(BlockDiagonal(V_cell)) * Matrix(BlockDiagonal(Theta_cell_true))
    b_0_true = -Matrix(BlockDiagonal(Vp_cell)) * vec(xsub)
    K        = size(b_0, 1)

    response_error = b_0_true - b_0
    F_obs_error = G_0*w_hat - G_0_true*w_hat
    F_obs_error[1]
    int_error = (RT/norm(RT)) \ (G_0_true*true_params - b_0_true)

    w_error_response = (RT/norm(RT))\G_0_true*(w_hat - true_params)

end

# -------------------------------------------------------
# applies a function f along dimension d of an array, treating each slice along that dimension as a vector input to f
# no attached functions
# -------------------------------------------------------
function arrayfunvec(arr_in::Array, f::Function, d::Int)
    return mapslices(f, arr_in, dims=d) # can convert to eachrow or eachcol later
end

# -------------------------------------------------------
# Builds a Jacobian matrix over all M time points for all features, uses symbolic variables
# no attached functions
# -------------------------------------------------------

function build_Jac_sym(features::Vector{Vector{Function}}, xobs::Matrix{Float64})
    M, nstates = size(xobs)
    flat_features = vcat(features...)
    J = length(flat_features)
    Jac_mat = zeros(J, nstates, M)

    @variables xs[1:nstates]
    xs_vec = collect(xs)

    for (j, f) in enumerate(flat_features)
        # pass xs_vec as a single vector argument
        f_sym  = f(xs_vec)
        g_sym  = Symbolics.gradient(f_sym, xs_vec)
        g_fns  = [Symbolics.build_function(g, xs_vec..., expression=Val{false})
                  for g in g_sym]
        for i in 1:M
            x_val = xobs[i, :]
            for state in 1:nstates
                Jac_mat[j, state, i] = g_fns[state](x_val...)
            end
        end
    end
    return Jac_mat
end

# -------------------------------------------------------
# takes a function phifun and returns its diff_order-th derivative 
# as a new callable function, using symbolic differentiation
# no attached functions
# -------------------------------------------------------
function dphi(phifun::Function, diff_order::Int)
    @variables y
    f_sym  = phifun(y)
    df_sym = Symbolics.derivative(f_sym, y, diff_order)
    return Symbolics.build_function(df_sym, y, expression=Val{false})
end

# -------------------------------------------------------
# Implements ensemble least squares and solves the least squares problem G/b multiple times 
# (for num runs) on random subsamples of rows
# then aggregates the results via mean or median
# no attached functions
# -------------------------------------------------------
function els(G::Matrix{Float64}, b::Vector{Float64},
    batch_size::Int, num_runs::Int, method::String)

    K, J = size(G)
    W    = zeros(J, num_runs)

    for rr in 1:num_runs
        inds     = randperm(K)[1 : floor(Int, K / batch_size)]
        W[:, rr] = G[inds, :] \ b[inds]
    end

    if method == "median"
        return median(W, dims=2) |> vec
    elseif method == "mean"
        return mean(W, dims=2) |> vec
    end
end

# -------------------------------------------------------
# estimate_sigma calculates the noise standard deviation in a noisy signal f
# with k = 6 on a symetric stencil to get 6th order finite difference filter
# normalizing the filter to it has unit L2 norm
# handles multi-dimensional arrays via conv
# -------------------------------------------------------
function estimate_sigma(f::VecOrMat{Float64}, k::Int=6, dim::Int=1)
    C    = fdcoeffF(k, 0.0, collect(Float64, -k-2 : k+2))
    filt = C ./ norm(C, 2)

    if dim > 1
        # permute target dim to front
        perm   = [dim; setdiff(1:ndims(f), dim)]
        f_perm = permutedims(f, perm)
        f_flat = reshape(f_perm, size(f_perm, 1), :)
        results = Float64[]
        for col in eachcol(f_flat)
            v = DSP.conv(col, filt)
            append!(results, v[length(filt) : end - length(filt) + 1])
        end
        return sqrt(mean(results .^ 2))
    else
        if ndims(f) == 1 || size(f, 2) == 1
            v     = DSP.conv(vec(f), filt)
            valid = v[length(filt) : end - length(filt) + 1]
            return sqrt(mean(valid .^ 2))
        else
            results = Float64[]
            for col in eachcol(f)
                v = DSP.conv(col, filt)
                append!(results, v[length(filt) : end - length(filt) + 1])
            end
            return sqrt(mean(results .^ 2))
        end
    end
end
# -------------------------------------------------------
# Calculates Fordberg finite difference weights (coefficients) for approximating the 
# k-th derivative of a function at point xbar, given a set of grid points x
# returns row vector c of dimension 1 by n, where n = length(x)
# For k = 0, this can be used to evaluate the interpolating polynomial itself
# Based on Fornberg weights algorithm
# -------------------------------------------------------
function fdcoeffF(k::Int, xbar::Float64, x::Vector{Float64})
    n = length(x)
    k >= n && error("length(x) must be larger than k")

    m  = k
    c1 = 1.0
    c4 = x[1] - xbar
    C  = zeros(n, m+1)
    C[1, 1] = 1.0

    for i in 1:n-1
        i1 = i + 1
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i1] - xbar
        for j in 0:i-1
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i - 1
                for s in mn:-1:1
                    s1 = s + 1
                    C[i1, s1] = c1 * (s * C[i1-1, s1-1] - c5 * C[i1-1, s1]) / c2
                end
                C[i1, 1] = -c1 * c5 * C[i1-1, 1] / c2
            end
            for s in mn:-1:1
                s1 = s + 1
                C[j1, s1] = (c4 * C[j1, s1] - s * C[j1, s1-1]) / c3
            end
            C[j1, 1] = c4 * C[j1, 1] / c3
        end
        c1 = c2
    end
    return C[:, end]  # return last column as vector
end


# -------------------------------------------------------
# analyzes the frequency content of observed data to find a corner point
# in the FFT (fast fourier transform) spectrum that separates the signal from the noise
# returns corner frequency and number of significant frequency components
# estimate of noise level based on the RMS of the low freq FFT
# needs getcorner()
# -------------------------------------------------------
function findcornerpts(xobs::VecOrMat{Float64}, t::Vector{Float64})
    t = vec(t)
    T = length(t)

    # wavenumbers centered at zero

    wn = collect((0:T-1) .- floor(T/2)) .* (2π / (maximum(t) - minimum(t)))
    xx = wn[1:ceil(Int, T/2)]
    NN = length(xx)

    # mean absolute FFT amplitude across columns (if matrix)
    Ufft = mean(abs.(fftshift(fft(xobs, 1)) ./ sqrt(2*NN)), dims=2)[:]
    Ufft = Ufft[1:ceil(Int, T/2)]

    # find peak frequency index
    _, Umax = findmax(Ufft)

    # find corner in cumulative sum up to peak
    tstarind = getcorner(cumsum(abs.(Ufft[1:Umax])), xx[1:Umax])

    tstar  = -xx[tstarind]
    k      = max(Umax - tstarind + 1, 1)
    corner = [tstar, Float64(k)]

    # noise estimate: RMS of FFT up to corner frequency
    sig_est = sqrt(mean(Ufft[1:min(Int(corner[2]), end)] .^ 2))

    return corner, sig_est
end


# -------------------------------------------------------
# determines the optimal test function support size (mt) and polynomial order (pts)
# for each state variable in the WENDy method
# avoids the need to manually tune the parameters
# phi class = 1, polynomial bump (1-x^2)^p
# phi class = 2, gaussian bump exp(-x^2)
# phi_class = function handle is custom test function
# -------------------------------------------------------
function findcorners(xobs::Matrix{Float64}, t::Vector{Float64},
    tau, tauhat, phi_class)

    T       = length(t)
    n       = size(xobs, 2)
    corners  = zeros(n, 2)
    sig_ests = zeros(n)
    mts      = zeros(Int, n)
    pts      = zeros(Float64, n)

    # expand scalar tauhat to vector
    if length(tauhat) == 1
        tauhat = fill(tauhat, n)
    end

    for nn in 1:n
        corner, sig_est = findcornerpts(xobs[:, nn], t)
        k = corner[2]

        if phi_class == 1
            # solve for m using fzero equivalent
            l = (m, k, N) -> log((2m - 1) / m^2) *
                (4π^2 * k^2 * m^2 - 3N^2 * tauhat[nn]^2) -
                2N^2 * tauhat[nn]^2 * log(tau)

            mnew = find_zero(m -> l(m, k, T), (1.0, 2.0 / sqrt(tau)))

            if mnew > T/2 - 1
                mnew = T / 2 / k
            end

        mts[nn] = min(floor(Int, (T-1)/2), ceil(Int, mnew))
        pts[nn] = max(2, floor(Int, log(tau) / log(1 - (1 - 1/mts[nn])^2)))

        elseif phi_class == 2
            mnew    = 1 + T * tauhat[nn] / 2π / k * sqrt(-2 * log(tau))
            mts[nn] = min(floor(Int, (T-1)/2), ceil(Int, mnew))
            pts[nn] = 2π * k / tauhat[nn] / T

        elseif phi_class isa Function
            mts[nn] = get_tf_support(phi_class, T, tauhat[nn], k)
            pts[nn] = NaN
        end

    corners[nn, :]  = corner
    sig_ests[nn]    = sig_est
    end

    return mts, pts, sig_ests, corners
end

# -------------------------------------------------------
# computes Jacobian of ODE RHS with respect ot the state variables x
# evaluated at all time points
# combines the learned parameters w with precomputed symbolic Jacobian Jac_max
# -------------------------------------------------------
function get_jac(Jac_mat::Array{Float64,3}, w::Vector{Float64},
    features::Vector{Vector{Function}})

    param_length_vec = [length(f) for f in features]

    # split w into chunks matching each equation
    w_cell = vec2cell(w, param_length_vec)

    # build block diagonal weight matrix
    w_cell_mat = [reshape(wc, :, 1) for wc in w_cell]
    W_blkdiag = Matrix(BlockDiagonal(w_cell_mat))'

    # pagemtimes: multiply W_blkdiag' × Jac_mat[:,:,i] for each page i
    _, nstates, M = size(Jac_mat)
    Jac = zeros(nstates, nstates, M)
    for i in 1:M
        Jac[:, :, i] = W_blkdiag * Jac_mat[:, :, i]
    end

    return Jac
end

# -------------------------------------------------------
# splits vector into chunks based on lengths
# -------------------------------------------------------
function vec2cell(w::Vector{Float64}, lengths::Vector{Int})
    chunks = Vector{Vector{Float64}}()
    idx = 1
    for l in lengths
        push!(chunks, reshape(w[idx : idx+l-1], l, 1) |> vec)
        idx += l
    end
    return chunks
end

# -------------------------------------------------------
# Builds two factors L0 and L1 that together represent the noise covariance structure
# Encode how covariance depends on both the test functions and the Jacobian of the feature library
# L0 represents baseline noise covariance from differentiating the data
# L1 is 3D array encoding how covariance changes with parameters w
# -------------------------------------------------------
function get_Lfac(Jac_mat::Array{Float64,3}, Js::Vector{Int},
    V_cell::Vector{Matrix{Float64}}, Vp_cell::Vector{Matrix{Float64}})
    _, d, M = size(Jac_mat)
    # permute to (d, M, J)
    Jac_mat = permutedims(Jac_mat, (2, 3, 1))
    eq_inds  = findall(x -> x > 0, Js)
    num_eq   = length(eq_inds)
    L0       = BlockDiagonal(Vp_cell)  # requires BlockDiagonals.jl
    L0_dense = Matrix(L0)
    total_K  = size(L0_dense, 1)
    total_J  = sum(Js)
    L1       = zeros(total_K, d*M, total_J)

    Ktot = 0; Jtot = 0
    for i in 1:num_eq
        K, _ = size(V_cell[i])
        J_i  = Js[eq_inds[i]]
        for ell in 1:d
            m = Jac_mat[ell, :, Jtot+1 : Jtot+J_i]'   # (J_i, M)
            n = V_cell[i]                               # (K, M)
            # L1[Ktot+1:Ktot+K, (ell-1)*M+1:ell*M, Jtot+1:Jtot+J_i]
            for ji in 1:J_i, ki in 1:K
                L1[Ktot+ki, (ell-1)*M+1 : ell*M, Jtot+ji] .= m[ji, :] .* n[ki, :]
            end
        end
        Ktot += K; Jtot += J_i
    end
    return L0_dense, L1
end

# -------------------------------------------------------
# general radius finder for tf phifun on [-1,1]
# Each method must depend on only one positive parameter p
# xobs must be a column vector so acts on each component of xobs
# -------------------------------------------------------
function get_rad(xobs::Vector{Float64}, tobs::Vector{Float64},
    phifun, method, p,
    mt_min::Int, mt_max::Int)

    if isnothing(phifun) || isnothing(method) || isnothing(p)
        return 0
    end

    if method == "direct"
        mt = p
    elseif method == "FFT"
        mt, _, _, _ = findcorners(xobs, tobs, nothing, p, phifun)
    elseif method == "timefrac"
        mt = floor(Int, length(tobs) * p)
    elseif method == "mtmin"
        mt = p * mt_min
    end

    return min(mt, mt_max)
end

# -------------------------------------------------------
# builds the Cholesky factor of the noise covariance matrix
# used in the IRLS loop to whiten the linear system
# lower triangular factor
# -------------------------------------------------------
function get_RT(L0::Matrix{Float64}, L1::Array{Float64,3},
    w::Vector{Float64}, diag_reg::Float64)

    dims = size(L1)  # (K, d*M, J)

    # update L0 if w is nonzero
    if !all(w .== 0)
        L1_perm    = permutedims(L1, (3, 1, 2))           # (J, K, d*M)
        L1_2d      = reshape(L1_perm, dims[3], :)          # (J, K*d*M)
        correction = reshape(L1_2d' * w, dims[1], :)       # (K, d*M)
        L0         = L0 .+ correction
    end

    # build regularized covariance
    Cov    = L0 * L0'
    n_ = size(Cov,1)
    newCov = (1 - diag_reg) .* Cov .+ diag_reg .* I(n_)

    RT = cholesky(newCov).L

return RT, L0, Cov, diag_reg
end

# -------------------------------------------------------
# builds polynomial approximation kernel of degree deg on a stencil of 2n + 1 points
# constructs a Vandermonde matr4ix X from the stensil points
# applies weighgint function k
# uses pseudoinverse to get coefficientsd A for each derivative up to max_dx
# returns selected derivative rows and full coef matrix 
# -------------------------------------------------------
function build_poly_kernel(deg::Int, k::Function, n::Int,
                           dx::Float64, max_dx::Int)
    x   = collect(-n:n) .* dx                        # stencil points
    X   = hcat([x .^ d for d in 0:deg]...)           # Vandermonde (2n+1 × deg+1)
    K   = k.(x ./ (n * dx))                          # weights
    K ./= norm(K, 1)
    sqK = sqrt.(K)
    A   = pinv(sqK .* X) .* sqK'                     # (deg+1 × 2n+1)

    # select derivative rows 0..max_dx
    fac = Diagonal(Float64[factorial(i) for i in 0:max_dx])
    M   = hcat(fac, zeros(max_dx+1, deg - max_dx))
    f   = M * A

    return f, A
end

# -------------------------------------------------------
# finds the optimal moving average filter width m for smoothing noisy data
# iteratively solves for m by balancing noise level with sigma_est 
# and signal curvature d
# iteration converges when m stops changing
# -------------------------------------------------------
function get_optimal_SMAF(
    x::Vector{Float64},
    fx_obs::VecOrMat{Float64};
    max_points::Int     = 10^5,
    init_m_fac::Int     = 200,
    max_filter_fac::Int = 8,
    expand_fac::Float64 = 2.0,
    maxits::Int         = 100,
    deriv_tol::Float64  = 1e-6,
    verbose::Bool       = false
)
    sigma_est = estimate_sigma(fx_obs, 6, 1)
    subsamp   = max(floor(Int, length(x) / max_points), 1)

    fx_subsamp = fx_obs[1:subsamp:end, :]
    M          = size(fx_subsamp, 1)
    dx_subsamp = mean(diff(x[1:subsamp:end]))

    m                = min(ceil(Int, M / init_m_fac),
                           floor(Int, M / max_filter_fac))
    max_filter_width = floor(Int, M / max_filter_fac)

    its = 1; check = 1

    while check > 0 && its < maxits
        # build kernel and get second-derivative row (index 3)
        n_kernel = min(max(floor(Int, m * expand_fac), 3),
                       floor(Int, (M-1)/2))
        _, A = build_poly_kernel(2, x -> fill(1.0, size(x)), n_kernel, dx_subsamp, 0)
        a_row = vec(A[3, :])   # second-derivative filter row

        # estimate curvature d
        if ndims(fx_subsamp) == 1 || size(fx_subsamp, 2) == 1
            v = DSP.conv(vec(fx_subsamp), a_row)
            valid = v[length(a_row) : end - length(a_row) + 1]
            d = 2 * mean(abs.(valid))
        else
            results = Float64[]
            for col in eachcol(fx_subsamp)
                v = DSP.conv(col, a_row)
                append!(results, abs.(v[length(a_row) : end - length(a_row) + 1]))
            end
            d = 2 * mean(results)
        end

        # solve n^5 - n^3 - C = 0 for optimal filter width
        C    = sigma_est^2 / ((d + deriv_tol)^2 * dx_subsamp^4 / 144)
        mnew = min(
            floor(Int, (find_zero(n -> n^5 - n^3 - C, 1.0) - 1) / 2),
            max_filter_width
        )

        check = abs(m - mnew)
        m     = mnew
        its  += 1

        if verbose
            println("iter=$its  m=$m  d=$d")
        end
    end

    m *= subsamp
    return m, sigma_est, its
end


# -------------------------------------------------------
# finds the minimum support size m for a custom test function phi
# best matches the target frequency
# defines target frequency energy representing where signals energy is concentrated
# iteratively increases support size m from 1 + 
# at each m, evaluates phi on grid of 2m + 1 points in [-1,1]
# zero pads to length N and takes FFT
# computes weighted frequency centroid of test function spectrum
# measures error between centroid and target
# stops when error increases
# -------------------------------------------------------
function get_tf_support(phi::Function, N::Int,
    tauhat, k_x::Float64)
    ks = ((0:N-1) .- floor(N/2)) .^ 2
    errs    = zeros(floor(Int, N/2) - 1)
    m       = 1
    target  = (k_x / tauhat)^2
    errs[1] = abs(target - sum(ks))
    check = false
    while !check && m <= (N-3)/2
        m += 1
        x        = -1 : (1/m) : 1
        phi_vals = phi.(x[2:end-1])                        # interior points
        phi_grid = [0.0; phi_vals; 0.0;                    # zero at endpoints
        zeros(N - 2m - 1)]                                  # zero padding to length N
        phi_fft  = fftshift(abs.(fft(phi_grid)))
        phi_fft ./= sum(phi_fft)
        errs[m]  = abs(target - sum(phi_fft .* ks))
        check = errs[m] > errs[m-1]
    end

    return m
end

# -------------------------------------------------------
# builds dense weight matrix from sparse representation of true parameter weights
# -------------------------------------------------------
function get_true_weights(weights::Vector, tags::Matrix, n::Int)

    true_nz_weights = zeros(size(tags, 1), n)

    for i in 1:length(weights)
        weights_i = weights[i]          # matrix: each row = [tags... value]
        l1, l2    = size(weights_i)

        for j in 1:l1
            tag_j = weights_i[j, 1:l2-1]   # feature tags for this row
            val_j = weights_i[j, l2]        # corresponding weight value

            # find which row of tags matches tag_j
            match = findfirst(k -> all(tags[k, :] .== tag_j), 1:size(tags, 1))

            if !isnothing(match)
                true_nz_weights[match, i] = val_j
            end
        end
    end

    return true_nz_weights
end

# -------------------------------------------------------
# builds test function matrix V using custom phifun and weights from phi_weights
# single-matrix version of getVVp 
# -------------------------------------------------------
function get_VVp_svd(mt::Int, t::Vector{Float64}, K::Int,
    phifun::Function, center_scheme)
    dt  = mean(diff(t))
    M   = length(t)
    # get test function weights — only need first row (function values)
    Cfs = phi_weights(phifun, mt, 1)
    v   = Cfs[1, :] .* dt                  # (2mt+1,) test function values
    if center_scheme == "uni"
        gap   = max(1, floor(Int, (M - 2mt) / K))
        diags = collect(0 : gap : M - 2mt - 1)
        diags = diags[1 : min(K, end)]
        V     = zeros(length(diags), M)
        for (j, _) in enumerate(diags)
            V[j, gap*(j-1)+1 : gap*(j-1)+2mt+1] .= v
        end

    elseif center_scheme == "random"
        gaps = randperm(M - 2mt)[1:K]
        V    = zeros(K, M)
        for j in 1:K
            V[j, gaps[j] : gaps[j]+2mt] .= v
        end
    elseif center_scheme isa Vector{Int}
        centers = unique(clamp.(center_scheme, mt+1, M-mt))
        K       = length(centers)
        V       = zeros(K, M)
        for (j, c) in enumerate(centers)
        V[j, c-mt : c+mt] .= v
        end
    end

    return V
end

# -------------------------------------------------------
# builds test function matrices V and Vp directly, either from custom phifun and 
# via phiweights or from finite difference coefficients if there is no phifun provided 
# -------------------------------------------------------
function get_VVp(mt::Int, t::Vector{Float64}, max_d::Int, K::Int,
    phifun, center_scheme)

    dt = mean(diff(t))
    M  = length(t)
    # --- compute weights ---
    if !isnothing(phifun)
        Cfs = phi_weights(phifun, mt, max_d)
    else
        # fallback: finite difference coefficients
        Cfs_mid      = fdcoeffF(1, t[mt], t[1:2mt-1])   # (2mt-1,) vector
        Cfs          = vcat([0.0 0.0],
                    Cfs_mid',
                    [0.0 0.0])'                  # (2mt+1 × 2) then transpose
        Cfs[2, :]  .*= -(mt * dt)                        # scale derivative row
    end
    # extract and scale test function and derivative rows
    v  = Cfs[end-1, :] .* (mt * dt)^(-max_d + 1) .* dt
    vp = Cfs[end,   :] .* (mt * dt)^(-max_d)      .* dt

    # --- place rows according to center scheme ---
    if center_scheme == "uni"
        gap   = max(1, floor(Int, (M - 2mt) / K))
        diags = collect(0 : gap : M - 2mt - 1)
        diags = diags[1 : min(K, length(diags))]
        V     = zeros(length(diags), M)
        Vp    = zeros(length(diags), M)
        for (j, _) in enumerate(diags)
            V[j,  gap*(j-1)+1 : gap*(j-1)+2mt+1] .= v
            Vp[j, gap*(j-1)+1 : gap*(j-1)+2mt+1] .= vp
        end
    elseif center_scheme == "random"
        gaps = randperm(M - 2mt)[1:K]
        V    = zeros(K, M)
        Vp   = zeros(K, M)
        for j in 1:K
            V[j,  gaps[j] : gaps[j]+2mt] .= v
            Vp[j, gaps[j] : gaps[j]+2mt] .= vp
        end

    elseif center_scheme isa Vector{Int}
        centers = unique(clamp.(center_scheme, mt+1, M-mt))
        K       = length(centers)
        V       = zeros(K, M)
        Vp      = zeros(K, M)
        for (j, c) in enumerate(centers)
            V[j,  c-mt : c+mt] .= v
            Vp[j, c-mt : c+mt] .= vp
        end
    end
    return V, Vp
end

# -------------------------------------------------------
# lin_regress
# -------------------------------------------------------
function lin_regress(U::Vector{Float64}, x::Vector{Float64})
    m = (U[end] - U[1]) / (x[end] - x[1])
    b = U[1] - m * x[1]
    L = U[1] .+ m .* (x .- x[1])
    return m, b, L
end

# -------------------------------------------------------
# build_lines
# -------------------------------------------------------
function build_lines(Ufft::Vector{Float64}, xx::Vector{Float64}, k::Int)
    NN       = length(Ufft)
    subinds1 = 1:k
    subinds2 = k:NN
    Ufft_av1 = Ufft[subinds1]
    Ufft_av2 = Ufft[subinds2]
    m1, b1, L1 = lin_regress(Ufft_av1, xx[subinds1])
    m2, b2, L2 = lin_regress(Ufft_av2, xx[subinds2])
    return L1, L2, m1, m2, b1, b2, Ufft_av1, Ufft_av2
end

# -------------------------------------------------------
# corner detection algorithm used to find elbow points in curves
# finds the index where a 1D curve best plits into 2 straight line segments
# adds argument to select the relative error (norm)
# -------------------------------------------------------
function getcorner(Ufft_in::Vector{Float64}, xx::Vector{Float64};
    norm_type::Symbol=:L2)
    NN   = length(Ufft_in)
    Ufft = Ufft_in ./ maximum(abs.(Ufft_in)) .* NN
    errs = zeros(NN)

    for k in 1:NN
        L1, L2, _, _, _, _, Uav1, Uav2 = build_lines(Ufft, xx, k)
        r1 = (L1 .- Uav1) ./ Uav1
        r2 = (L2 .- Uav2) ./ Uav2
        if norm_type == :L1
            errs[k] = sum(abs.(r1)) + sum(abs.(r2))
        elseif norm_type == :L2
            errs[k] = sqrt(sum(r1 .^ 2) + sum(r2 .^ 2))
        elseif norm_type == :L2sq
            errs[k] = sum(r1 .^ 2) + sum(r2 .^ 2)
        end
    end

    errs[isnan.(errs)] .= Inf
    return argmin(errs)
end


# -------------------------------------------------------
# blowup detection step that stops ODE integration if the solution blows up above thresh
# -------------------------------------------------------
function make_blowup_callback(thresh::Float64)
    condition  = (u, t, integrator) -> norm(u, Inf) - thresh
    affect!    = (integrator) -> terminate!(integrator)
    return ContinuousCallback(condition, affect!)
end

# -------------------------------------------------------
# integrate ODE with given params using Rodas4 if stiff and Tsit5 for non-stiff
# -------------------------------------------------------
function get_sol(params, rhs::Function, tspan,
                 x0::Vector{Float64}, tol_ode::Float64,
                 odemethod::String, thresh::Float64)

    # wrap rhs to inject params
    ode! = (du, u, p, t) -> begin
        du .= rhs(u, params)
    end

    prob = ODEProblem(ode!, x0, (tspan[1], tspan[end]))
    cb   = make_blowup_callback(thresh)

    if odemethod == "ode15s"
        # ode15s from matlab ~ Rodas4 or QNDF (stiff solvers)
        sol = solve(prob, Rodas4(),
                    saveat=tspan,
                    reltol=tol_ode,
                    abstol=tol_ode,
                    callback=cb)
    else
        # ode45 from matlab ~ Tsit5 (non-stiff)
        sol = solve(prob, Tsit5(),
                    saveat=tspan,
                    reltol=tol_ode,
                    abstol=tol_ode,
                    callback=cb)
    end

    return Array(sol)'   # (time × states)
end

# -------------------------------------------------------
# objective function simulates ODE with current params then computes
# squared Frobenius like norm of the residual between simulated and observed data
# optimizer minimizes to find the best parameters
# -------------------------------------------------------
function NLS_compare(xobs::Matrix{Float64}, params,
                     rhs_pv::Function, tobs::Vector{Float64},
                     x0_nls::Vector{Float64}, tol_ode_nls::Float64,
                     odemethod_nls::String, thresh::Float64)

    yobs = get_sol(params, rhs_pv, tobs, x0_nls, tol_ode_nls, odemethod_nls, thresh)

    # align sizes in case of early termination
    n_common = size(yobs, 1)
    residual = xobs[1:n_common, :] .- yobs

    # norm(vecnorm(residual))^2
    col_norms = [norm(residual[:, j]) for j in 1:size(residual, 2)]
    return norm(col_norms)^2
end

# -------------------------------------------------------
# output selector utility function, calls a function f with input
# and returns the nth output argument
# -------------------------------------------------------
function outn(f::Function, input, n::Int)
    return f(input)[n]
end

# -------------------------------------------------------
# computes matrix of test function values and derivatives evaluated on uniform grid
# used to build quadrature weights for test function matrices V and Vp
# -------------------------------------------------------
function phi_weights(phifun::Function, m::Int, maxd::Int)

    # creates uniform grid of 2m + 1 on [-1,1]
    xf   = collect(range(-1.0, 1.0, length=2m+1)) 
    x    = xf[2:end-1]                          # interior points only
    Cfs  = zeros(maxd+1, 2m+1)

    @variables y
    f_sym = phifun(y)

    for j in 1:maxd+1
        # symbolic derivative of order j-1
        Df_sym = f_sym
        for _ in 1:j-1
            Df_sym = Symbolics.derivative(Df_sym, y)
        end
        Df_fn  = Symbolics.build_function(Df_sym, y, expression=Val{false})

        # evaluate on interior points
        vals = map(xi -> begin
            v = Df_fn(xi)
            isnan(v) ? Df_fn(xi + sign(xi) * eps(Float64)) : v
        end, x)

        Cfs[j, 2:end-1] .= vals

        # fix any infinities
        for k in 1:2m+1
            if isinf(abs(Cfs[j, k]))
                xk = xf[k]
                Cfs[j, k] = Df_fn(xk - sign(xk) * eps(Float64))
            end
        end
    end

    # normalize by L2 norm of zeroth derivative row
    Cfs ./= norm(Cfs[1, :], 2)

    return Cfs
end

# -------------------------------------------------------
# adaptive radius selector for test functions
# finds optimal support size mt by minimizing spectral aliasining error across candidate radii
# supports upsampling and proximity weighting
# -------------------------------------------------------

function rad_select(t0::Vector{Float64}, y::VecOrMat{Float64},
    phifun, inc::Int, sub::Float64, q::Float64,
    s::Float64, m_min::Int, m_max::Int, pow)

    # --- early exits ---
    if isnothing(phifun)
        return m_min
    end
    if m_max <= m_min
        return m_min
    end

    M, nstates = size(y, 1), size(y, 2)
    dt  = mean(diff(t0))
    t   = collect(0 : dt/inc : (M-1)*dt)

    # --- proximity-weighted FFT ---
    if q > 0
        prox_u     = t -> exp.(-abs.(t .- t[floor(Int, length(t)/2)]) .^ q)
        prox_u_vec = (dt/inc/sqrt(M*dt)) .* fftshift(fft(prox_u(t)))
    else
        if inc > 1
        # spline interpolation onto upsampled grid
            interp_cols = hcat([begin
            itp = cubic_spline_interpolation(t0, y[:, nn], extrapolation_bc=Line())
            itp.(t)
        end for nn in 1:nstates]...)
            prox_u_vec = (dt/inc/sqrt(M*dt)) .* fftshift(fft(interp_cols, 1))
        elseif inc == 1
            prox_u_vec = (dt/sqrt(M*dt)) .* fftshift(fft(y, 1))
        end
    end

    errs = Float64[]
    ms   = Int[]

    # --- main loop over candidate radii ---
    for m in m_min:m_max
        t_phi     = range(-1 + dt/inc, 1 - dt/inc, length=2*inc*m - 1)
        Qs        = collect(1 : floor(Int, s*inc*m) : length(t) - 2*inc*m)
        errs_temp = zeros(nstates, length(Qs))

        for (Qi, Q) in enumerate(Qs)
            # place test function at center Q
            phi_vec         = zeros(length(t))
            phi_len         = length(t_phi)
            phi_vec[Q : Q + phi_len - 1] .= phifun.(collect(t_phi))
            phi_vec       ./= norm(phi_vec, 2)

            for nn in 1:nstates
                phiu_fft = (dt / sqrt(M*dt)) .* fft(phi_vec .* y[:, nn])
                n_alias  = floor(Int, inc*M/2)
                step     = floor(Int, M/sub)
                alias    = phiu_fft[1 : step : n_alias]
                errs_temp[nn, Qi] = 2 * (2π / sqrt(M*dt)) *
                                    sum((0:length(alias)-1) .* imag.(alias))
            end
        end

        push!(errs, sqrt(mean(errs_temp .^ 2)))   # rms
        push!(ms,   m)
    end

    # --- select optimal radius ---
    if pow isa String
        # findchangepts equivalent: use changepoint detection
        b = _findchangept_log(errs)
        #b = _findchangept_log(errs, pow) uses PELT algorithm which is slightly different from matlab 
        if isnothing(b)
            b = argmin(errs .* sqrt.(ms))
        end
    elseif isnothing(pow)
        b = getcorner(log.(errs), Float64.(ms))
    else
        b = argmin(errs .* ms .^ pow)
    end

    return ms[b]
end

# -------------------------------------------------------
# imple changepoint detection on log(errs)
# -------------------------------------------------------
function _findchangept_log(errs::Vector{Float64})
    log_errs = log.(errs)
    NN       = length(log_errs)
    costs    = zeros(NN)
    for k in 2:NN-1
        seg1  = log_errs[1:k]
        seg2  = log_errs[k+1:end]
        costs[k] = var(seg1) * length(seg1) + var(seg2) * length(seg2)
    end
    costs[1] = Inf; costs[end] = Inf
    b = argmin(costs)
    return b == 1 ? nothing : b
end

# -------------------------------------------------------
# helper function that uses Changepoints.jl instead of approximation above
# -------------------------------------------------------
# using Changepoints
# function _findchangept_log(errs::Vector{Float64}, statistic::String="mean")
#     log_errs = log.(errs)
    
#     if statistic == "linear"
#         # linear trend changepoint — closest to MATLAB's 'linear' statistic
#         cpts, _ = @PELT log_errs Normal() 1
#     elseif statistic == "mean"
#         # mean shift detection
#         cpts, _ = @PELT log_errs Normal(0.0, 1.0) 1
#     elseif statistic == "std"
#         # variance changepoint
#         cpts, _ = @PELT log_errs Normal(1.0, 0.0) 1
#     else
#         # default fallback
#         cpts, _ = @PELT log_errs Normal() 1
#     end

#     if isempty(cpts)
#         return nothing
#     else
#         return cpts[1]   # return first changepoint, matching MATLAB's b(1)
#     end
# end

# -------------------------------------------------------
# SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
# [H, pValue, SWstatistic] = SWTEST(X, ALPHA) performs the
# Shapiro-Wilk test to determine if the null hypothesis of
# composite normality is a reasonable assumption regarding the
# population distribution of a random sample X. The desired significance 
# level, alpha, is an optional scalar input (default = 0.05).
# -------------------------------------------------------
function swtest(x::Vector{Float64}, alpha::Float64=0.05)

    # --- input checks ---
    x = filter(!isnan, x)

    if length(x) < 3
        @warn "Sample vector must have at least 3 valid observations."
        return nothing, 0.0, nothing
    end
    if length(x) > 5000
        @warn "Shapiro-Wilk test might be inaccurate for n > 5000."
    end
    if !(0 < alpha < 1)
        error("Significance level alpha must be between 0 and 1.")
    end

    # --- setup ---
    x       = sort(x)
    n       = length(x)
    mtilde  = quantile.(Normal(), ((1:n) .- 3/8) ./ (n + 1/4))
    weights = zeros(n)

    # --- branch on kurtosis ---
    if kurtosis(x) > 3

        # ---- Shapiro-Francia test (leptokurtic) ----
        weights = mtilde ./ sqrt(dot(mtilde, mtilde))
        W       = (dot(weights, x))^2 / sum((x .- mean(x)).^2)

        # Royston (1993a, p.183)
        nu  = log(n)
        u1  = log(nu) - nu
        u2  = log(nu) + 2/nu
        mu  = -1.2725 + 1.0521 * u1
        sig = 1.0308  - 0.26758 * u2

        newSFstat      = log(1 - W)
        NormalSFstat   = (newSFstat - mu) / sig
        pValue         = 1 - cdf(Normal(), NormalSFstat)

    else

        # ---- Shapiro-Wilk test (platykurtic) ----
        c = mtilde ./ sqrt(dot(mtilde, mtilde))
        u = 1 / sqrt(n)

        # polynomial coefficients (Royston 1992, 1993b)
        PolyCoef_1 = [-2.706056,  4.434685, -2.071190, -0.147981,  0.221157, c[n]  ]
        PolyCoef_2 = [-3.582633,  5.682633, -1.752461, -0.293762,  0.042981, c[n-1]]
        PolyCoef_3 = [-0.0006714, 0.0250540, -0.39978,  0.54400]
        PolyCoef_4 = [-0.0020322, 0.0627670, -0.77857,  1.38220]
        PolyCoef_5 = [ 0.00389150,-0.083751, -0.31082, -1.5861 ]
        PolyCoef_6 = [ 0.00303020,-0.082676, -0.48030]
        PolyCoef_7 = [ 0.459, -2.273]

        weights[n] = evalpoly(u, reverse(PolyCoef_1))
        weights[1] = -weights[n]

        if n > 5
            weights[n-1] = evalpoly(u, reverse(PolyCoef_2))
            weights[2]   = -weights[n-1]
            count = 3
            phi   = (dot(mtilde, mtilde) - 2*mtilde[n]^2 - 2*mtilde[n-1]^2) /
                    (1 - 2*weights[n]^2 - 2*weights[n-1]^2)
        else
            count = 2
            phi   = (dot(mtilde, mtilde) - 2*mtilde[n]^2) /
                    (1 - 2*weights[n]^2)
        end

        # special case n=3
        if n == 3
            weights[1] =  1/sqrt(2)
            weights[n] = -weights[1]
            phi = 1.0
        end

        weights[count : n-count+1] .= mtilde[count : n-count+1] ./ sqrt(phi)

        W    = (dot(weights, x))^2 / sum((x .- mean(x)).^2)
        newn = log(n)

        if 4 <= n <= 11
            mu             = evalpoly(Float64(n), reverse(PolyCoef_3))
            sig            = exp(evalpoly(Float64(n), reverse(PolyCoef_4)))
            gam            = evalpoly(Float64(n), reverse(PolyCoef_7))
            newSWstat      = -log(gam - log(1 - W))

        elseif n > 11
            mu             = evalpoly(newn, reverse(PolyCoef_5))
            sig            = exp(evalpoly(newn, reverse(PolyCoef_6)))
            newSWstat      = log(1 - W)

        else  # n <= 3
            mu             = 0.0
            sig            = 1.0
            newSWstat      = 0.0
        end

        NormalSWstat = (newSWstat - mu) / sig
        pValue       = 1 - cdf(Normal(), NormalSWstat)

        # special case n=3
        if n == 3
            pValue = 6/π * (asin(sqrt(W)) - asin(sqrt(3/4)))
        end

    end

    H = pValue < alpha ? 1 : 0
    return H, pValue, W
end
# -------------------------------------------------------
# vector splitter takes flat vector w and splits into cell array of sub-vectors
# specified by length_vec
# -------------------------------------------------------

function vec2cell(w::Vector{Float64}, length_vec::Vector{Int})
    stops  = cumsum(length_vec)
    starts = [1; stops[1:end-1] .+ 1]
    return [w[starts[nn] : stops[nn]] for nn in eachindex(length_vec)]
end

# -------------------------------------------------------
# computes singular value decomposition (SVD)-compressed test function matrices V and Vp
# Vp is obtained through FFT differentiation rather than through direct computation
# orthogonal test functions that are more numerically stable for large systems than get_VVp
# -------------------------------------------------------
function VVp_svd(V::Matrix{Float64}, K_min::Int,
                 t::Vector{Float64}, toggle_VVp_svd::Float64)

    m  = length(t)
    dt = mean(diff(t))

    # economy SVD of V'
    U, s_vec, _ = svd(V', full=false)
    sings        = s_vec

    # --- rank selection ---
    if toggle_VVp_svd > 0
        # energy threshold
        cum_energy = cumsum(sings .^ 2) ./ sum(sings .^ 2)
        s = findfirst(x -> x > toggle_VVp_svd^2, cum_energy)
        if isnothing(s)
            s = min(size(V, 1), size(V, 1))   # fallback: keep all
        end
    else
        # corner detection on cumulative singular value sum
        corner_data = cumsum(sings) ./ sum(sings)
        xx          = collect(Float64, 1:length(corner_data))
        s           = getcorner(corner_data, xx, norm_type=:L1)
        s           = min(max(K_min, s), size(V, 1))
    end

    # --- build compressed V ---
    inds = 1:s
    V    = U[:, inds]' .* dt                 # (s × m)

    # --- FFT differentiation for Vp ---
    Vp     = copy(V')                        # (m × s)
    Vp_hat = fft(Vp, 1)                     # FFT along time dimension

    # build wavenumber vector
    if mod(m, 2) == 0
        k = Float64[0:m÷2; (-m÷2+1):-1]
    else
        k = Float64[0:floor(Int, m/2); (-floor(Int, m/2)):-1]
    end

    # spectral derivative: multiply by (2πi/m/dt)*k
    Vp_hat .*= (2π / m / dt) .* im .* k

    # zero Nyquist for even m to avoid asymmetry
    if mod(m, 2) == 0
        Vp_hat[m÷2, :] .= 0
    end

    Vp = real(ifft(Vp_hat, 1))'             # (s × m)

    return V, Vp
end

function wendy_opt(G::Matrix{Float64}, b::VecOrMat{Float64},
                   method::String     = "LS",
                   batch_size::Int  = 1,
                   num_runs::Int    = 1,
                   avg_method::String = "mean",
                   cov              = nothing)

    if method == "LS"
        if isnothing(cov)
            # standard least squares
            w = G \ b
        else
            # weighted least squares: min (Gw-b)' * inv(cov) * (Gw-b)
            C   = cholesky(cov).L
            w   = (C \ G) \ (C \ b)
        end

    elseif method == "TLS"
        # total least squares via SVD of augmented matrix [G b]
        _, n    = size(G)
        _, _, V = svd(hcat(G, b), full=false)
        w       = -V[1:n, n+1:end] ./ V[n+1, n+1]

    elseif method == "ensLS"
        # ensemble least squares
        w = els(G, vec(b), batch_size, num_runs, avg_method)
    end

    return w
end

function wendy_fcn_0(
    xobs::Matrix{Float64},
    tobs::Vector{Float64},
    features::Vector{Vector{Function}},
    toggle_smooth::Int,
    mt_cell::Matrix,                        # cm × cn matrix of param tuples
    mt_min::Int,
    mt_max::Int,
    K_min::Int,
    K_max::Int,
    center_scheme,
    toggle_VVp_svd::Float64,
    w0::Union{Vector{Float64}, Nothing},
    optim_params,
    iter_diff_tol::Float64,
    max_iter::Int,
    diag_reg::Float64,
    pvalmin::Float64,
    check_pval_it::Int
)

    # -------------------------------------------------------
    # dimensions
    # -------------------------------------------------------
    M, nstates       = size(xobs)
    param_length_vec = [length(f) for f in features]
    eq_inds          = findall(f -> !isempty(f), features)
    num_eq           = length(eq_inds)

    # -------------------------------------------------------
    # estimate noise variance and build initial covariance
    # -------------------------------------------------------
    sig_ests = [estimate_sigma(xobs[:, i]) for i in 1:nstates]
    RT_0     = spdiagm(0 => repeat(sig_ests, inner=M))

    # -------------------------------------------------------
    # optional smoothing
    # -------------------------------------------------------
    if toggle_smooth > 0
        expand_fac = 1.5
        sws        = zeros(Int, nstates)
        RT_0_temps = SparseMatrixCSC[]

        for nn in 1:nstates
            if toggle_smooth == 1
                sws[nn], _, _ = get_optimal_SMAF(
                    tobs, xobs[:, nn],
                    expand_fac=expand_fac
                )
            else
                sws[nn] = toggle_smooth
            end

            # moving average smoothing
            hw = sws[nn]
            for i in 1:M
                lo = max(1, i - hw)
                hi = min(M, i + hw)
                xobs[i, nn] = mean(xobs[lo:hi, nn])
            end

            # banded smoothing matrix
            bw      = sws[nn]
            RT_temp = spdiagm([k => ones(M - abs(k))
                               for k in -bw:bw]...)
            RT_temp = RT_temp ./ vec(sum(RT_temp, dims=2))
            push!(RT_0_temps, RT_temp)
        end

        RT_0 = Matrix(BlockDiagonal(RT_0_temps)) * RT_0
    end

    # -------------------------------------------------------
    # build test function matrices
    # -------------------------------------------------------
    cm, cn = size(mt_cell)
    K      = min(floor(Int, K_max / nstates / cm), M)

    V_cell  = Vector{Matrix{Float64}}(undef, num_eq)
    Vp_cell = Vector{Matrix{Float64}}(undef, num_eq)

    if cn < nstates
        # shared radius across states — harmonic mean over scales
        mt = [get_rad(xobs[:, i], tobs,
                      mt_cell[j, 1][1], mt_cell[j, 1][2],
                      mt_cell[j, 1][3], mt_min, mt_max)
              for j in 1:cm, i in 1:nstates]
        mt = ceil.(Int, 1  ./ mean(1 ./ mt, dims=2))

        if toggle_VVp_svd != 0 && cm > 1
            # SVD path
            V_all = vcat([get_VVp_svd(mt[j], tobs, K,
                                       mt_cell[j, 1][1], center_scheme)
                          for j in 1:cm]...)
            V, Vp = VVp_svd(V_all, K_min, tobs, toggle_VVp_svd)
        else
            # direct path
            VVp_pairs = [get_VVp(mt[j], tobs, 1, K,
                                  mt_cell[j, 1][1], center_scheme)
                         for j in 1:cm]
            V  = vcat([p[1] for p in VVp_pairs]...)
            Vp = vcat([p[2] for p in VVp_pairs]...)
        end

        V_cell  = [V  for _ in 1:num_eq]
        Vp_cell = [Matrix(Vp) for _ in 1:num_eq]
        mt      = repeat(mt, 1, nstates)

    else
        # per-state radius
        mt = [get_rad(xobs[:, min(nn, nstates)], tobs,
                      mt_cell[j, min(nn, cn)][1],
                      mt_cell[j, min(nn, cn)][2],
                      mt_cell[j, min(nn, cn)][3],
                      mt_min, mt_max)
              for j in 1:cm, nn in 1:nstates]

        for nn in 1:num_eq
            V_nn  = Matrix{Float64}[]
            Vp_nn = Matrix{Float64}[]
            active = findall(x -> x > 0, mt[:, nn])

            for j in active
                cf_idx = min(nn, size(mt_cell, 2))
                if toggle_VVp_svd != 0 && cm > 1
                    push!(V_nn, get_VVp_svd(mt[j, nn], tobs, K,
                                             mt_cell[j, cf_idx][1],
                                             center_scheme))
                else
                    V_j, Vp_j = get_VVp(mt[j, nn], tobs, 1, K,
                                          mt_cell[j, cf_idx][1],
                                          center_scheme)
                    push!(V_nn,  V_j)
                    push!(Vp_nn, Vp_j)
                end
            end

            if toggle_VVp_svd != 0 && cm > 1
                V_cell[nn], Vp_cell[nn] = VVp_svd(
                    vcat(V_nn...), K_min, tobs, toggle_VVp_svd)
            else
                V_cell[nn]  = vcat(V_nn...)
                Vp_cell[nn] = vcat(Vp_nn...)
            end
        end
    end

    # -------------------------------------------------------
    # build linear system
    # -------------------------------------------------------
    xobs_cell  = [xobs[:, i] for i in 1:nstates]
    Theta_cell = [begin
                cols = [[f(xobs[i, :]) for i in 1:M] for f in flist]
                hcat(cols...)   # (M × num_features)
                end
                for flist in features]

    G_0 = Matrix(BlockDiagonal([V * T
                             for (V, T) in zip(V_cell, Theta_cell)]))
    b_0 = vcat([-Vp * x
                for (Vp, x) in zip(Vp_cell, xobs_cell)]...)

    # -------------------------------------------------------
    # build Jacobian and L-factors
    # -------------------------------------------------------
    Jac_mat = build_Jac_sym(features, xobs)
    L0 = nothing; L1 = nothing
    if max_iter > 1
        L0, L1 = get_Lfac(Jac_mat, param_length_vec, V_cell, Vp_cell)
        L0      = L0 * Matrix(RT_0)
        for i in axes(L1, 3)
            L1[:, :, i] = L1[:, :, i] * Matrix(RT_0)
        end
    end

    # -------------------------------------------------------
    # initialize
    # -------------------------------------------------------
    w_hat     = isnothing(w0) ? wendy_opt(G_0, b_0, optim_params) : w0
    w_hat_its = copy(w_hat)                          # will grow as matrix
    res       = G_0 * w_hat .- b_0                  # initial residual
    res_0     = copy(res)
    iter      = 1; check = 1.0; pval = 1.0
    RT        = Matrix(I, size(b_0, 1), size(b_0, 1))
    _, pvals, _ = swtest(vec(res))
    pvals     = [pvals]

    # -------------------------------------------------------
    # IRLS loop
    # -------------------------------------------------------
    while check > iter_diff_tol && iter < max_iter && pval > pvalmin

        # update covariance
        RT, _, _, _ = get_RT(L0, L1, vec(w_hat), diag_reg)
        G = RT \ G_0
        b = RT \ b_0

        # update parameters
        w_hat = wendy_opt(G, b, optim_params)
        res_n = G * w_hat .- b

        # check stopping conditions
        _, pval_new, _ = swtest(vec(res_n))
        push!(pvals, pval_new)
        if iter + 1 > check_pval_it
            pval = pvals[end]
        end

        check = norm(w_hat_its[:, end] .- w_hat) /
                norm(w_hat_its[:, end])
        iter += 1

        # accumulate results
        res       = hcat(res,   res_n)
        res_0     = hcat(res_0, G_0 * w_hat .- b_0)
        w_hat_its = hcat(w_hat_its, w_hat)
    end

    # -------------------------------------------------------
    # handle divergence
    # -------------------------------------------------------
    if pval < pvalmin
        println("Warning: WENDy iterates diverged")
        idx   = argmax(pvals)
        w_hat = w_hat_its[:, idx]

        if idx == 1
            RT = Matrix(I, size(b_0, 1), size(b_0, 1))
        elseif idx < size(w_hat_its, 2)
            RT, _, _, _ = get_RT(L0, L1, w_hat_its[:, idx-1], diag_reg)
        end

        res       = hcat(res,   res[:, idx])
        res_0     = hcat(res_0, res_0[:, idx])
        w_hat_its = hcat(w_hat_its, w_hat_its[:, idx])
    end

    # -------------------------------------------------------
    # uncertainty quantification
    # -------------------------------------------------------
    Ginv = G_0 \ RT
    CovW = Ginv * Ginv'
    stdW = sqrt.(diag(CovW))
    mseW = mean(res[:, end] .^ 2)

    return (w_hat, res, res_0, w_hat_its,
            V_cell, Vp_cell, Theta_cell, mt,
            xobs, Jac_mat, G_0, b_0, RT,
            stdW, mseW, CovW)
end



function display_wendy_results()

    
end

function display_wendy_results()

    err_norm = 2
    errs = [norm(w_hat_its[:, i] .- true_params, err_norm) / norm(true_params, err_norm)
            for i in 1:size(w_hat_its, 2)] # euclidean norm
    err_wendy = last(errs)
    println("err WENDy: "*string(err_wendy))
    println("parameters: true, WENDy")
    hcat(w_hat, true_params)
    G = RT \ G_0
    b = RT\ b_0
    res_true = G*true_params-b
    res_0_true = G_0*true_params - b_0
    err_NLS = 0
    total_time_nls = 0
    param_nls = nothing

    println("-----------------")
    println("")
    println("time elapsed: "*string(total_time)*" seconds")
    println("# iterations: "*string(size(w_hat_its)[1]))
    println("K, d, M: ", string(size(V_cell[1])[1]), ", ", string(nstates), ", ", string(length(tobs)))
    println("test function radii: "*string(length(unique(vec(mt)))))
    println("")
    println("-----------------")

    x_cell = [xsub[:, i] for i in 1:nstates]

    Theta_cell_true = [hcat([[f(xsub[i, :]) for i in 1:M] for f in flist]...)
                    for flist in features]

    G_0_true = Matrix(BlockDiagonal(V_cell)) * Matrix(BlockDiagonal(Theta_cell_true))
    b_0_true = -Matrix(BlockDiagonal(Vp_cell)) * vec(xsub)
    K        = size(b_0, 1)

    response_error = b_0_true - b_0
    F_obs_error = G_0*w_hat - G_0_true*w_hat
    F_obs_error[1]
    int_error = (RT/norm(RT)) \ (G_0_true*true_params - b_0_true)

    w_error_response = (RT/opnorm(RT))\G_0_true*(w_hat - true_params)

    J = get_jac(Jac_mat, w_hat,features)

    noise = xobs - xsub
    noise_perm      = reshape(permutedims(noise, (2, 1)), nstates, 1, M)
    page_result     = stack([J[:, :, i] * noise_perm[:, :, i]
                            for i in 1:size(J, 3)], dims=3)
    flat            = vec(dropdims(page_result, dims = 2)')  # transpose each page then flatten
    Jac_F_obs_error = Matrix(BlockDiagonal(V_cell)) * flat
    lin_approx = (RT/opnorm(RT)) \ (Jac_F_obs_error + response_error)

    nonlin_res = (RT/opnorm(RT)) \ (F_obs_error - Jac_F_obs_error)

    Res_full = w_error_response + int_error + lin_approx + nonlin_res
    
    # add plot

    tau = 10^-10
    tauhat = 1
    _,_,_,corners = findcorners(xobs,tobs,tau,tauhat,phifun)

    xfft = abs.(fft(xobs, 1))
    xfft = xfft./maximum(xfft, dims = 1)
    phifft = stack([abs.(fft(vec(Vp_cell[i][1, :]))) for i in 1:length(Vp_cell)])'
    phifft = phifft./maximum(phifft,dims = 2)

    xflip = collect(vcat(1:length(w_hat), reverse(1:length(w_hat))))
    c = 0.05
    stdW = max.(sqrt.(diag(CovW)),eps(Float64))

    conf_int   = quantile.(Normal(0, 1), 1 - c/2) .* stdW
    confbounds = [w_hat .- conf_int; reverse(w_hat .+ conf_int)]
    err_all    = [norm(col .- true_params) for col in eachcol([true_params w_hat])] ./ norm(true_params)

    pvals   = [swtest(res[:, i])[2]   for i in 1:size(res,   2)]
    pvals_0 = [swtest(res_0[:, i])[2] for i in 1:size(res_0, 2)]

    p1 = plot(1:length(errs),errs, yscale=:log10, marker = :circle, 
        linecolor = :blue, markercolor = :white, markerstrokecolor = :blue,
        label = L"err(\textbf{w^n})",
        title = "err(OLS)="*string(round(errs[1]*100; digits = 4))*", err(WENDy)="*string(round(errs[length(errs)]*100; digits = 4)),
        ylabel = L"||\textbf{w^n - w^* ||_2~/~||w^*||_2}", xlabel = "iter",
        ylims = (10^log10(minimum(errs))*0.9,10^log10(maximum(errs))*1.2));

    p2 = scatter(1:length(w_hat), w_hat, markercolor = :white, markershape = :circle, markersize = 1, markerstrokecolor = :red,
        label = L"\mathrm{w_{WENDy}}", title = "95% confidence bounds",xlim = (0, length(w_hat)+1));

    ymin = ifelse(minimum(confbounds) <0, minimum(confbounds)*1.2, minimum(confbounds)*0.9)
    ymax = ifelse(maximum(confbounds) <0, maximum(confbounds)*0.9, maximum(confbounds)*1.2)
    for j in 1:length(w_hat)
        x_min = j - 0.25
        x_max = j + 0.25
        y_min = w_hat[j] - conf_int[j]
        y_max = w_hat[j] + conf_int[j]
        x_coords = [x_min, x_max, x_max, x_min, x_min]
        y_coords = [y_max, y_max, y_min, y_min, y_max]
        plot!(p2, x_coords,y_coords, markersize = 1, seriestype=:shape, opacity = 0.3,
        label = "", linecolor =:black, fillcolor = :lightblue, fillalpha = 0);
        plot!(p2, [x_min,x_max],[w_hat[j], w_hat[j]], seriestype=:line, linecolor =:gray, label = "");
    end
    scatter!(p2, 1:length(w_hat), true_params, markercolor = :blue, markershape = :xcross,
        label = L"\mathrm{w^*}", ylim = (ymin,ymax));


    p3 = plot(pvals, marker = :circle, linecolor = :blue, label = L"p-val(w^n)", 
    markercolor = :white, markerstrokecolor = :blue,
        title = "p-val(OLS)="*string(round(pvals_0[1], digits = 12))*", p-val(WENDy)="*string(round(pvals[length(pvals)], digits = 4)))


    labels = reshape(repeat([""], nstates),1,nstates)    
    labels[1] = L"u^*"
    p4 = plot(tobs, xsub, label = labels, alpha = 0.8, linewidth = 5, color = :black,
        title = "data: (K,d,M)=("*string(K)*","*string(nstates)*","*string(M)*")");
    labels[1] = "U"
    scatter!(p4, tobs, xobs, label = labels, markersize = 3, alpha = 0.8, markerstrokewidth=0, color = :red);
    err_dd = round(norm(xsub - xtest)/norm(xsub)*100, digits = 4)
    labels[1] = L"U_{dd}:"*string(err_dd)*"% rel. err"
    plot!(p4, ttest, xtest,label = labels, linewidth = 3, linestyle = :dash, color = :chartreuse);

    p5 = plot(opnorm(RT)*res[:,size(res)[2]], color =:blue, linewidth = 5, label = L"W_{WENDy}",
        title = L"\mathrm{C^{1/2}r(U,w). p-val (WENDy) = } "*string(round(pvals[length(pvals)], digits = 4)));
    plot!(p5, opnorm(RT)*res_true, color = :red, linewidth = 5, linestyle = :dash, label = L"w^*");

    p6 = plot(Res_full, color =:black, linewidth = 5, label = L"r(U,W)",
        title = L"C^{1/2}r(U,w)~vs.~C^{1/2}L_w\epsilon");
    plot!(p6, lin_approx, color = :red, linewidth = 5, linestyle = :dash, label = L"L_we");

    labels[1] = "F(U)"
    p7 = plot(xfft[1:Int(floor(size(xfft)[1]/2)),:], yscale=:log10, color = :red, label = labels,
        title = "Fourier content of data",
        xlabel = "wavenumber", ylim = (minimum(xfft)/10,1));
    labels[1] = L"F(\Phi(1,:)))"
    plot!(p7, phifft[:,1:Int(floor(size(phifft)[2]/2))]', yscale=:log10, color = :cyan1, label = labels);

    p8 = plot(res_0[:,1], color = :blue, label = "w_{WENDy}",
    title = L"r(U,W). p-val(OLS)="*string(round(pvals_0[1], digits = 12)),
        ylim = maximum(abs.(res[:,1]))*[-1.5,1.5]);
    plot!(p8, res_0_true, color = :red, linestyle = :dash);

    p9 = plot(w_error_response, color = :blue, linewidth = 5, label = L"r_0", linestyle = :dash,
        ylim = maximum(abs.(Res_full))*[-1.2,1.2],
        title = "other residual components");
    plot!(p9, int_error, color = :red, linewidth = 5, label = L"e_{int}", linestyle = :dash);
    plot!(p9, nonlin_res, color = :yellow, linewidth = 5, label = "h", linestyle = :dash);

    #savefig("figures/WENDyComp_SIR3.pdf")
    plot(p1,p2,p3,p4,p5,p6,p7,p8,p9, layout = (3,3), size = (2000, 1200))
end