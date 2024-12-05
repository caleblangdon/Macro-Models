using LinearAlgebra
using Distributions
using Plots
using QuantEcon

mutable struct Hopenhayn1992
    β
    B
    α
    c_f
    c_e
    μ_s
    ρ
    σ_ϵ
    Nz
    display_plots
    price_ss
    tol
    maxit
    F
    prod_grid
    G
    VF
    firm_profit
    firm_output
    pol_n
    continue_indicator
    exit_cutoff
    gamma_at_1
    Y
    m_star
    gamma
    total_mass
    pdf_stationary
    cdf_stationary
    distrib_emp
    pdf_emp
    cdf_emp
    total_employment
    average_firm_size
    average_entrant_size
    relative_size_of_new_entrants
    exit_rate
    report_dict
end

function Hopenhayn1992(;
    β=0.96,
    B=100,
    α=2//3,
    c_f=15,
    c_e=50,
    μ_s=1.2,
    ρ=0.9,
    σ_ϵ=0.2,
    Nz=20,
    display_plots=true,
    tol=1e-8,
    maxit=2000,
    )

    model = Hopenhayn1992(
    β,
    B,
    α,
    c_f,
    c_e,
    μ_s,
    ρ,
    σ_ϵ,
    Nz,
    display_plots,
    nothing,
    tol,
    maxit,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    )
    setup_grid!(model)
    return model
end

function setup_grid!(model::Hopenhayn1992)
    # Discretely approximate the continuous AR(1) process
    mc = rouwenhorst(model.Nz, model.ρ, model.σ_ϵ, model.μ_s * (1 - model.ρ))
    model.F = mc.p
    model.prod_grid = exp.(mc.state_values)
    model.G = stationary_distributions(mc)[1]
end

function static_profit_max(model::Hopenhayn1992, price)
    optimal_n = (model.α * price .* model.prod_grid) .^ (1 / (1 - model.α))
    firm_output = model.prod_grid .* (optimal_n .^ model.α)
    firm_profit = price .* firm_output .- optimal_n .- price .* model.c_f
    return firm_profit, firm_output, optimal_n
end

function incumbent_firm(model::Hopenhayn1992, price)
    VF_old = zeros(model.Nz)
    VF = zeros(model.Nz)
    firm_profit, firm_output, pol_n = static_profit_max(model, price)
    for it in 1:model.maxit
        VF = firm_profit .+ model.β .* max.(model.F * VF_old, 0)
        dist = maximum(abs.(VF_old - VF))
        if dist < model.tol
            break
        end
        VF_old .= VF
    end
    continue_indicator = (model.F * VF .>= 0)
    idx = searchsortedfirst(continue_indicator, true)
    exit_cutoff = model.prod_grid[idx]
    return VF, firm_profit, firm_output, pol_n, continue_indicator, exit_cutoff
end

function find_equilibrium_price(model::Hopenhayn1992)
    pmin, pmax = 0.0, 100.0
    price = 0.0
    for it_p in 1:model.maxit
        price = (pmin + pmax) / 2
        VF = incumbent_firm(model, price)[1]
        VF_entrant = dot(VF, model.G)
        diff = abs(VF_entrant - (price * model.c_e))
        if diff < model.tol
            break
        end
        if VF_entrant < price * model.c_e
            pmin = price
        else
            pmax = price
        end
    end
    return price
end

function solve_invariant_distribution(model::Hopenhayn1992, m, continue_indicator)
    F_tilde = (model.F .* continue_indicator')'
    I_matrix  = I(model.Nz)
    return m .* (inv(I_matrix  - F_tilde) * model.G)
end

function track_cohort(model::Hopenhayn1992, n)
    exit_rate = zeros(n)
    relative_size = zeros(n + 1)
    current_dist = model.G
    current_size = model.m_star
    current_gamma = current_dist .* current_size

    for age in 0:n
        if age > 0
            current_gamma = model.F' * (current_gamma .* model.continue_indicator)
            current_size = sum(current_gamma)
            current_dist = current_gamma ./ current_size
        end

        current_average_size = dot(model.pol_n, current_dist)
        relative_size[age + 1] = current_average_size / model.average_firm_size
        current_continue_rate = dot(current_dist, model.continue_indicator)

        if age > 0
            exit_rate[age] = 1 - current_continue_rate
        end

        println("Age: $age")
        if age > 0
            println("Exit rate: $(exit_rate[age])")
        end
        println("Relative firm size: $(relative_size[age + 1])")
    end

    if model.display_plots
        fig = plot(layout = (1, 2), size = (1000, 500))
    
        # First plot
        plot!(fig[1], 1:n, exit_rate, label = "Exit Rate")
        yformatter = y -> string(round(y * 100, digits = 2)) * "%"
        ylims!(fig[1], 0, 1)
        # yticks!(fig[1], 0:0.1:1)
        # yticklabels!(fig[1], yformatter.(0:0.1:1))
    
        # Second plot
        plot!(fig[2], 0:n, relative_size, label = "Relative Size")
    
        display(fig)
    end
end

function solve_model!(model::Hopenhayn1992)
    t0 = time()
    model.price_ss = find_equilibrium_price(model)
    model.VF, model.firm_profit, model.firm_output, model.pol_n, model.continue_indicator, model.exit_cutoff = incumbent_firm(model, model.price_ss)
    model.gamma_at_1 = solve_invariant_distribution(model, 1, model.continue_indicator)
    model.Y = model.B / model.price_ss
    model.m_star = model.Y / (dot(model.gamma_at_1, model.firm_output .- model.c_f) - model.c_e)
    model.gamma = model.m_star .* model.gamma_at_1
    model.total_mass = sum(model.gamma)
    model.pdf_stationary = model.gamma ./ model.total_mass
    model.cdf_stationary = cumsum(model.pdf_stationary)
    model.distrib_emp = model.pol_n .* model.gamma
    model.pdf_emp = model.distrib_emp ./ sum(model.distrib_emp)
    model.cdf_emp = cumsum(model.pdf_emp)
    model.total_employment = dot(model.pol_n, model.gamma)
    model.average_firm_size = model.total_employment / model.total_mass
    model.average_entrant_size = dot(model.pol_n, model.G)
    model.relative_size_of_new_entrants = model.average_entrant_size / model.average_firm_size
    model.exit_rate = model.m_star / model.total_mass
    model.report_dict = Dict(
        "Steady State Equilibrium Price" => model.price_ss,
        "Entry Mass" => model.m_star,
        "Mass of all firms" => model.total_mass,
        "Firm entry/exit rate" => model.exit_rate,
        "Average Firm Size" => model.average_firm_size,
        "Relative Size of New Entrants" => model.relative_size_of_new_entrants,
        "Aggregate Output" => model.Y,
        "Aggregate Employment" => model.total_employment
    )
    println("\n-----------------------------------------------------------")
    for (string, var) in model.report_dict
        println("$string: $var")
    end
    println("-----------------------------------------------------------")
    track_cohort(model, 5)
    if model.display_plots
        p = plot(model.prod_grid, model.VF, label="Value Function")
        vline!(p, [model.exit_cutoff], label="Exit Threshold")
        hline!(p, [0], label="Zero Line")
        title!(p, "Incumbent Firm Value Function")
        xlabel!(p, "Productivity level")
        display(p)
    end
    t1 = time()
    println("\nTotal Run Time: $(t1 - t0) seconds")
end

# model = Hopenhayn1992()
# @time solve_model!(model)