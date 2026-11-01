# Public entry points for the dependency-free sampler. Split from
# static_hmc.jl only to keep each file a readable length; included immediately
# after it.

"""
    static_nuts(gradf!, x0, n_draws, n_warmup; delta=0.8, seed=1)

NUTS with a diagonal metric, adapting the step size towards acceptance rate
`delta` and the metric during `n_warmup` iterations. Written without
dependencies so that it compiles under `juliac --trim`.

`gradf!(g, x)` must write the gradient of the log density at `x` into `g` and
return the log density. Warm-up draws are discarded. Returns a
[`StaticHMCResult`](@ref).
"""
function static_nuts(gradf!::F, x0::Vector{Float64}, n_draws::Int,
                     n_warmup::Int; delta::Float64=0.8, seed::Int=1) where {F}
    d = length(x0)
    rng = _Rng(seed)
    x = copy(x0)
    g = Vector{Float64}(undef, d)
    lp = gradf!(g, x)

    inv_mass = ones(d)
    eps = _find_initial_eps(x, gradf!, inv_mass, rng)
    da = _DualAverage(eps, delta)

    # Stan's windowed adaptation: an initial fast interval for the step size
    # alone, then doubling windows that re-estimate the metric, then a final
    # fast interval to settle the step size against the last metric.
    wel = _Welford(d)
    win_start = min(75, max(1, n_warmup ÷ 10))
    win_end = max(n_warmup - 50, win_start + 1)
    next_window = min(win_start + 25, win_end)

    draws = Matrix{Float64}(undef, d, n_draws)
    lps = Vector{Float64}(undef, n_draws)

    n_div = 0; acc_sum = 0.0; acc_n = 0; depth_sum = 0
    n_div_warm = 0; acc_sum_warm = 0.0; acc_n_warm = 0

    p = Vector{Float64}(undef, d)

    for it in 1:(n_warmup + n_draws)
        warm = it <= n_warmup

        @inbounds for i in 1:d
            p[i] = _rand_normal(rng) / sqrt(inv_mass[i])
        end
        h0 = -lp + _kinetic(p, inv_mass)
        log_u = log(_rand_uniform(rng)) - h0

        x_minus = copy(x); p_minus = copy(p); g_minus = copy(g)
        x_plus = copy(x);  p_plus = copy(p);  g_plus = copy(g)
        x_prop = copy(x);  lp_prop = lp
        log_weight = -h0
        # Zero, NOT copy(p): every subtree returns the summed momentum of the
        # states it VISITED, and the starting state is not one of them. Seeding
        # with p double-counts the initial momentum, which corrupts the U-turn
        # test and truncates trajectories -- it showed up as draws ~26% too
        # dispersed on a standard normal, with mean tree depth stuck below 1.
        p_sum = zeros(d)
        sum_alpha = 0.0; n_alpha = 0
        diverged = false
        depth = 0

        while depth < _STATIC_MAX_DEPTH
            direction = _rand_uniform(rng) < 0.5 ? -1 : 1
            sub = if direction == -1
                _build_tree(x_minus, p_minus, g_minus, log_u, depth, direction,
                            eps, inv_mass, h0, gradf!, rng)
            else
                _build_tree(x_plus, p_plus, g_plus, log_u, depth, direction,
                            eps, inv_mass, h0, gradf!, rng)
            end

            if sub.diverged
                diverged = true
                sum_alpha += sub.sum_alpha; n_alpha += sub.n_alpha
                break
            end

            # Accept the subtree's proposal with probability w_sub/(w_old+w_sub).
            if sub.log_weight > -Inf &&
               log(_rand_uniform(rng)) < sub.log_weight - log_weight
                copyto!(x_prop, sub.x_prop)
                lp_prop = sub.lp_prop
            end
            log_weight = _logaddexp(log_weight, sub.log_weight)
            sum_alpha += sub.sum_alpha; n_alpha += sub.n_alpha

            if direction == -1
                x_minus = sub.x_minus; p_minus = sub.p_minus; g_minus = sub.g_minus
            else
                x_plus = sub.x_plus; p_plus = sub.p_plus; g_plus = sub.g_plus
            end
            @inbounds for i in 1:d
                p_sum[i] += sub.p_sum[i]
            end

            sub.stopped && break
            _no_u_turn(p_sum, p_minus, p_plus, inv_mass) || break
            depth += 1
        end

        copyto!(x, x_prop)
        lp = gradf!(g, x)

        a_bar = n_alpha > 0 ? sum_alpha / n_alpha : 0.0

        if warm
            diverged && (n_div_warm += 1)
            acc_sum_warm += a_bar; acc_n_warm += 1
            eps = _adapt!(da, a_bar)
            if it > win_start && it <= win_end
                _accumulate!(wel, x)
                if it == next_window
                    inv_mass = _variance(wel)
                    _reset!(wel)
                    eps = _find_initial_eps(x, gradf!, inv_mass, rng)
                    da = _DualAverage(eps, delta)
                    next_window = min(it + 2 * (it - win_start), win_end)
                end
            end
            it == n_warmup && (eps = _final_eps(da))
        else
            diverged && (n_div += 1)
            acc_sum += a_bar; acc_n += 1
            depth_sum += depth
            k = it - n_warmup
            @inbounds for i in 1:d
                draws[i, k] = x[i]
            end
            lps[k] = lp
        end
    end

    return StaticHMCResult(draws, lps, eps, inv_mass, n_div,
                           acc_n > 0 ? acc_sum / acc_n : 0.0,
                           n_div_warm,
                           acc_n_warm > 0 ? acc_sum_warm / acc_n_warm : 0.0,
                           acc_n > 0 ? depth_sum / acc_n : 0.0)
end

"""
    static_hmc(gradf!, x0, inv_mass, eps, L, n_draws, n_warmup; seed=1, jitter=true)

HMC with a fixed diagonal inverse metric `inv_mass`, step size `eps` and up to
`L` leapfrog steps, written without dependencies so that it compiles under
`juliac --trim`. Nothing is adapted, so tune `eps` and `inv_mass` beforehand,
for example with [`static_nuts`](@ref).

`gradf!` is as for [`static_nuts`](@ref). The first `n_warmup` draws are
discarded. With `jitter=true` the step count is drawn from `1:L` each
iteration; `jitter=false` always takes `L`. Returns a
[`StaticHMCResult`](@ref).
"""
function static_hmc(gradf!::F, x0::Vector{Float64}, inv_mass::Vector{Float64},
                    eps::Float64, L::Int, n_draws::Int, n_warmup::Int;
                    seed::Int=1, jitter::Bool=true) where {F}
    d = length(x0)
    length(inv_mass) == d ||
        throw(ArgumentError("inv_mass has length $(length(inv_mass)), expected $d"))
    all(>(0.0), inv_mass) ||
        throw(ArgumentError("inv_mass entries must be positive"))
    L >= 1 || throw(ArgumentError("L must be at least 1"))

    rng = _Rng(seed)
    x = copy(x0)
    g = Vector{Float64}(undef, d)
    lp = gradf!(g, x)

    p = Vector{Float64}(undef, d)
    x_save = Vector{Float64}(undef, d)
    g_save = Vector{Float64}(undef, d)
    draws = Matrix{Float64}(undef, d, n_draws)
    lps = Vector{Float64}(undef, n_draws)

    n_div = 0; acc_sum = 0.0; acc_n = 0

    for it in 1:(n_warmup + n_draws)
        warm = it <= n_warmup

        @inbounds for i in 1:d
            p[i] = _rand_normal(rng) / sqrt(inv_mass[i])
        end
        h0 = -lp + _kinetic(p, inv_mass)

        copyto!(x_save, x); copyto!(g_save, g); lp_save = lp

        # Drawn independently of the current state, which is what keeps the
        # usual Metropolis correction on the end point valid (Neal 2011 sec. 4.2).
        if jitter
            steps = 1 + Int(floor(_rand_uniform(rng) * L))
            steps > L && (steps = L)
        else
            steps = L
        end

        diverged = false
        for _ in 1:steps
            lp = _leapfrog!(x, p, g, eps, inv_mass, gradf!)
            if !isfinite(lp)
                diverged = true
                break
            end
        end

        h1 = diverged ? Inf : -lp + _kinetic(p, inv_mass)
        dh = h0 - h1
        (!isfinite(dh) || dh < -_STATIC_DELTA_MAX) && (diverged = true)

        a = diverged ? 0.0 : (dh > 0.0 ? 1.0 : exp(dh))
        if !(!diverged && _rand_uniform(rng) < a)
            copyto!(x, x_save); copyto!(g, g_save); lp = lp_save
        end

        if !warm
            diverged && (n_div += 1)
            acc_sum += a; acc_n += 1
            k = it - n_warmup
            @inbounds for i in 1:d
                draws[i, k] = x[i]
            end
            lps[k] = lp
        end
    end

    return StaticHMCResult(draws, lps, eps, inv_mass, n_div,
                           acc_n > 0 ? acc_sum / acc_n : 0.0, 0, 0.0, 0.0)
end

"""
    static_gradient(ldf) -> gradf!

Wrap a [`LogDensityFunction`](@ref) with a gradient as the `gradf!(g, x)`
function the static samplers take. Useful for running them from Julia; a
trimmed binary should supply its own gradient instead.
"""
function static_gradient(ldf)
    return function (g::Vector{Float64}, x::Vector{Float64})
        lp, grad = LogDensityProblems.logdensity_and_gradient(ldf, x)
        @inbounds for i in eachindex(g)
            g[i] = grad[i]
        end
        return lp
    end
end
