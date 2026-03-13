#WendyODEs

struct odeModel
    features::Vector{Vector{Function}}
    params::Vector{Vector{Float64}}
end

function (m::odeModel)(dx, x, p, t)
    for i in eachindex(x)
        dx[i] = sum(f(x) * c for (f, c) in zip(m.features[i], m.params[i]))
    end
end

function simulate(model::odeModel, x0, tspan; tol=1e-12, n=501)
    t = range(tspan[1], tspan[2], length=n)
    prob = ODEProblem(model, x0, tspan)
    sol = solve(prob, Tsit5(), saveat=t, reltol=tol, abstol=tol)
    return Array(sol)', sol.t
end

function gen_noise(U_exact, sigma_NR, noise_dist, noise_alg)
    if noise_alg == 0  # additive
        stdv = mean(U_exact .^ 2) # the square of root mean squared
    elseif noise_alg == 1  # multiplicative
        stdv = 1.0
    end

    dims = size(U_exact)

    if noise_dist == 0  # white noise
        if sigma_NR > 0
            sigma = sigma_NR * sqrt(stdv)
        else
            sigma = -sigma_NR
        end
        noise = randn(dims...) .* sigma # random normal dist with sd sigma and mean 0 
    elseif noise_dist == 1  # uniform noise
        if sigma_NR > 0
            sigma = sqrt(3 * sigma_NR^2 * stdv)
        else
            sigma = -sigma_NR
        end
        noise = sigma .* (2 .* rand(dims...) .- 1)
    end

    if noise_alg == 0  # additive
        U = U_exact .+ noise
    elseif noise_alg == 1  # multiplicative
        U = U_exact .* (1 .+ noise)
    end

    noise_ratio_obs = norm(U - U_exact) / norm(U_exact) # norm calculates the Euclidean 2-norm
    return U, noise, noise_ratio_obs, sigma
end

a = 10
b = 28
c = 8/3
lorenz_model = odeModel(
    [
        [x -> x[2], x -> x[1]],
        [x -> x[1], x -> x[1]*x[3], x -> x[2]],
        [x -> x[1]*x[2], x -> x[3]]
    ],
    [[a, -a], [b, -1.0, -1.0], [1.0, -c]]
)

r = 4.91 # prey growth rate
b = 0.753 # attack rate
c = 0.022 # conversion rate
d = 0.089 # predator death rate
lotka_volterra_model = odeModel(
    [
        [x -> x[1], x -> x[1]*x[2]],
        [x -> x[2], x -> x[1]*x[2]]
    ],
    [[r, -b], [-d, c, -1.0]]
)

r = 2.0
k = -0.25
logistic_model = odeModel(
    [[x -> x[1], x -> x[1]^2]],
    [[r, -k]]
)

beta = 0.4
gamma = 0.2
delta = 0.1
seir_model = odeModel(
    [
        [x -> x[1]*x[3]],
        [x -> x[1]*x[3], x -> x[2]],
        [x -> x[2], x -> x[3]],
        [x -> x[3]]
    ],
    [[-beta], [beta, -gamma], [gamma, -delta], [delta]]
)

beta1 = 0.55
gamma1 = 0.15
delta1 = 0.1
sir_model = odeModel(
    [
        [x -> x[1]*x[2], x-> x[3]],
        [x -> x[1]*x[2], x -> x[2]],
        [x -> x[2], x -> x[3]]
    ],
    [[-beta1, delta1], [beta1, -gamma1], [gamma1, -delta1]]
)


kf = 0.1 # kinetic forward
kr = 0.1 # kinetic referse
kcat = 0.1 # catalytic binding
michaelis_menten_model = odeModel(
    [
        [x -> x[1]*x[2], x-> x[3]], # substrate
        [x -> x[1]*x[2], x -> x[3], x -> x[3]], # enzyme
        [x -> x[1]*x[2], x -> x[3], x -> x[3]], # complex
        [x -> x[3]] # product
    ],
    [[-kf, kr], [-kf, kr, kcat], [kf, -kr,  -kcat], [kcat]]
)


function sim_model(mod; tmax = 10.0, tout = 501::Int64, subsample = 1::Int64, noise_rat = 0.0::Float64)

    if mod == "lorenz"
        xobs, tobs = simulate(lorenz_model, [-8.0, 10.0, 27.0], (0.0, tmax), n = tout)
    elseif mod == "seir"
        xobs, tobs = simulate(seir_model, [1.0, 0, 0.01, 0.0], (0.0, tmax), n = tout)
    elseif mod == "logistic"
        xobs, tobs = simulate(logistic_model, [0.01], (0.0, tmax), n = tout)
    elseif mod =="lotka-volterra"
        xobs, tobs = simulate(lotka_volterra_model, [1, 1], (0.0, tmax), n = tout)
    elseif mod == "sir"
        xobs, tobs = simulate(sir_model, [0.99,0.01,0], (0.0,tmax), n = tout)
    elseif mod == "michaelis-menten"
        xobs, tobs = simulate(michaelis_menten_model, [200.0,10.0,0.0,0.0], (0.0,tmax), n = tout)
    end

    if subsample > 1
        keep_ids = 1:subsample:length(tobs)
        tsub = tobs[keep_ids]
        xsub = xobs[keep_ids,:]
    else 
        tsub = tobs
        xsub = xobs
    end

    if noise_rat > 0.0
        Random.seed!(11)
        noise_ratio = noise_rat
        noise_dist = 0
        noise_alg = 0

        xobs_n, noise, _, sigma = gen_noise(xsub, noise_ratio, noise_dist, noise_alg)
        xsub = xobs_n
    else 
        xsub = xsub
    end

    return(xsub,tsub)
end

function get_features(mod)
    # features_log = [[x -> x[1], x -> x[1]^2]]
    # features_seir = [
    #         [x -> x[1]*x[3]],
    #         [x -> x[1]*x[3], x -> x[2]],
    #         [x -> x[2], x -> x[3]],
    #         [x -> x[3]]
    #     ]
    # features_lorenz = [[x -> x[2], x -> x[1]],
    #         [x -> x[1], x -> x[1]*x[3], x -> x[2]],
    #         [x -> x[1]*x[2], x -> x[3]]]
    # features_lv = [
    #         [x -> x[1], x -> x[1]*x[2]],
    #         [x -> x[2], x -> x[1]*x[2]]
    #     ]
    # features_sir = [
    #     [x -> x[1]*x[2], x-> x[3]],
    #     [x -> x[1]*x[2], x -> x[2]],
    #     [x -> x[2], x -> x[3]]
    # ]
    # features_mm = [
    #     [x -> x[1]*x[2], x-> x[3]], # substrate
    #     [x -> x[1]*x[2], x -> x[3], x -> x[3]], # enzyme
    #     [x -> x[1]*x[2], x -> x[3], x -> x[3]], # complex
    #     [x -> x[3]] # product
    # ]
    
    if mod == "lorenz"
        features = lorenz_model.features
        params = lorenz_model.params
    elseif mod == "seir"
        features = seir_model.features
        params = seir_model.params    
    elseif mod == "logistic"
        features = logistic_model.features
        params = logistic_model.params
    elseif mod =="lotka-volterra"
        features = lotka_volterra_model.features
        params = lotka_volterra_model.params
    elseif mod == "sir"
        features = sir_model.features
        params = sir_model.params
    elseif mod == "michaelis-menten"
        features = michaelis_menten_model.features
        params = michaelis_menten_model.params
    end     

    return(features, params)
end


