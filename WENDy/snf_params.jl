#snf_params

# smooth data
toggle_smooth = 0

# set optimization params
optim_params = "LS"

# set weak integration
eta = 9
phi_funs = [
    x -> exp.(-eta .* (1 .- x.^2).^(-1)),
    x -> (1 .- x.^2).^eta
]
center_scheme = "uni"
toggle_VVp_svd = 0.99
submt = 2.1

# set jacobian correction params
max_iter = 100
iter_diff_tol = 10^-10
err_norm = 2
diag_reg = 10^-10
check_pval_it = 10
pvalmin = 10^-4
w0 = nothing
