using DifferentialEquations 
using CSV
#using DataFrames
using Distributions 
using LinearAlgebra 
using Tables
#using Dates
using Random 
using Plots
using LaTeXStrings 
using ColorSchemes

cd("/Users/katherinedixon/Documents/StuffINeed/_Research/wendy/julia/")

include("./WENDyAlg3.jl")
include("./snf_params.jl")
include("./WendyODEs.jl")

models = ["lorenz","seir","logistic","lotka-volterra","sir", "michaelis-menten"]

mod_pick = "sir"

#xobs, tobs = sim_model(mod_pick, tmax = 80.0, tout = 321, subsample = 0, noise_rat = 0.1)

xobs = CSV.read("../matlab/wendy/output/matlab_sir_obs.csv", Tables.matrix; header = false)
tobs = CSV.read("../matlab/wendy/output/matlab_sir_time.csv", Tables.matrix; header = false)[:,1]
xsub = CSV.read("../matlab/wendy/output/matlab_sir_obs_true.csv", Tables.matrix; header = false)
w_hat_ml = CSV.read("../matlab/wendy/output/matlab_sir_what.csv", Tables.matrix; header = false)

true_model = get_features(mod_pick)
true_params = reduce(vcat, true_model[2])

features = true_model[1]

M, nstates = size(xobs)

function run_wendy(xobs, tobs, features)

    M, nstates = size(xobs)

    K_max = 5000
    K_min = length(vcat(features...))
    phifun = phi_funs[1]
    method = "mtmin"
    mt_params = 2 .^(0:3)
    mt_max = Int64(max(floor((M-1)/2)-K_min,1))

    mt_min = rad_select(tobs,xobs, phifun, 1,submt, 0.0,1.0,2,mt_max,nothing)
    mt_cell = reshape([(phifun, method, p) for p in mt_params], length(mt_params), 1)

    t1 = time()
    w_hat, res, res_0, w_hat_its,
            V_cell, Vp_cell, Theta_cell, mt,
            xobs, Jac_mat, G_0, b_0, RT,
            stdW, mseW, CovW = wendy_fcn_0(xobs,tobs, features, toggle_smooth, mt_cell, mt_min,
            mt_max, K_min, K_max, center_scheme, 
    toggle_VVp_svd, w0, optim_params, iter_diff_tol,
    max_iter, diag_reg, pvalmin, check_pval_it)
    t2 = time() - t1

    return w_hat, res, res_0, w_hat_its,
            V_cell, Vp_cell, Theta_cell, mt,
            xobs, Jac_mat, G_0, b_0, RT,
            stdW, mseW, CovW,t2
end

size(V_cell)


#IRSL_SL_model = WENDy(features)
#w_hat = fit_IRLS(IRSL_SL_model, xobs_n, tsub, radius = nothing, gap = 1, type_rad = 0, p = 16, S = 1, mu = [1,2,1], toggle_SVD = false, diag_reg = 1e-10, trunc = 0)
#model = IRSL_SL_model

w_hat, res, res_0, w_hat_its,
        V_cell, Vp_cell, Theta_cell, mt,
        xobs, Jac_mat, G_0, b_0, RT,
        stdW, mseW, CovW, total_time = run_wendy(xobs,tobs, features)

w_hat


true_params_vv = true_model[2]

n = 1
pred_params = Vector{Vector{Float64}}(undef, 3)
for i in 1:size(true_params_vv)[1]
        p = reduce(vcat, true_params_vv[i,:])
        v = w_hat[n:(n + length(p)-1)]

        n = n + length(p)
        pred_params[i] = v
end


model_test = odeModel(
    features,
    [[w_hat[1], w_hat[2]], [w_hat[3], w_hat[4]], [w_hat[5], w_hat[6]]]
)

xtest, ttest = simulate(model_test, [0.99,0.01,0.0], (0.0, 80.0), n = 321)

display_wendy_results()

savefig("figures/SIR_data_pred.pdf")
scatter(tobs, xobs, label = ["S_noise" "I_noise" "R_noise"], markersize = 3, alpha = 0.8, markerstrokewidth=0, color = [:green :red  :blue])
plot!(tobs, xsub, label = ["S_actual" "I_actual" "R_actual"],  alpha = 0.8, markerstrokewidth=0, linewidth = 5, color = [:green :red  :blue])

plot!(ttest, xtest,label = ["S_pred" "I_pred" "R_pred"], linewidth = 5, linestyle = :dot, color = [:green3 :lightcoral :deepskyblue3])