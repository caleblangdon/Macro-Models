using Distributions, Statistics
using QuadGK, Roots
using Plots

struct Burdett1978
    δ::Float64
    s::Float64
    λ::Float64
    β::Float64
    r::Float64
    b::Float64
    Nw::Int
    low_lim::Float64
    up_lim::Float64
    wage_dist::Distribution
    f::Function
    F::Function
    w_grid::Vector{Float64}
    f_grid::Vector{Float64}
    F_grid::Vector{Float64}
    w_R::Float64
    g::Function
    G::Function
    g_grid::Vector{Float64}
    G_grid::Vector{Float64}
    max_iter::Int
    tol::Float64
    verbose::Bool
end

function Burdett1978(;
    δ::Float64 = 0.03,
    s::Float64 = 0.5,
    λ::Float64 = 0.4,
    β::Float64 = 0.995,
    Nw::Int = 200,
    μ::Float64 = 1.0,
    σ::Float64 = sqrt(0.1),
    low_lim::Float64 = 0.0,
    up_lim::Float64 = 3.5,
    max_iter::Int = 10_000,
    tol::Float64 = 1e-5,
    verbose::Bool = true)

    r = -log(β)
    wage_dist = truncated(LogNormal(μ,σ), low_lim, up_lim)
    f(w) = pdf(wage_dist, w)
    F(w) = cdf(wage_dist, w)
    avg_wage_offer, _ = quadgk(x -> x * f(x), low_lim, up_lim)
    b = 0.4 * avg_wage_offer
    
    w_R = solve_reservation_wage(δ,s,λ,r,b,low_lim,up_lim,F)
    if verbose
        println("Model initialized with reservation wage $w_R")
    end
        
    G(w) = w < w_R ? 0.0 : (δ * (F(w) - F(w_R))) / ((1 - F(w_R)) * (s * λ * (1 - F(w)) + δ))
    g(w) = w < w_R ? 0.0 : (f(w) * δ * (δ + s * λ * (1 - F(w)) + s * λ * (F(w) - F(w_R)))) / ((1 - F(w_R)) * (δ + s * λ * (1 - F(w)))^2)
        
    w_01 = quantile(wage_dist, 0.01)
    w_99 = quantile(wage_dist, 0.99)
    w_grid = range(w_01, w_99, length=Nw)
    f_grid = f.(w_grid) / sum(f.(w_grid))
    F_grid = cumsum(f_grid)
    g_grid = g.(w_grid) / sum(g.(w_grid))
    G_grid = cumsum(g_grid)
    
    return Burdett1978(δ,s,λ,β,r,b,Nw,low_lim,up_lim,wage_dist,f,F,w_grid,f_grid,F_grid,w_R,g,G,g_grid,G_grid,max_iter,tol,verbose)
end

function solve_reservation_wage(δ,s,λ,r,b,low_lim,up_lim,F)
    integrand(w) = (1-F(w))/(r+δ+s*λ*(1-F(w)))
    function equation(w_R)
        integral, _ = quadgk(x -> integrand(x), w_R, up_lim)
        return w_R - b - λ*(1-s)*integral
    end
    root = find_zero(equation, (low_lim, up_lim), Bisection())
    return root
end

model = Burdett1978()