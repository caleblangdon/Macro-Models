# imports
import time
import numpy as np
import scipy as sp
import quantecon as qe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
# from numba import njit, prange
sns.set_theme(style='whitegrid')

class HR93:
    """
    Class object of the Hopenhayn and Rogerson (1993) model.
    """


    ##############
    # Initiation #
    ##############


    def __init__(self,
                 β = 0.96,      # discount factor
                 B = 100,       # demand primitive
                 α = 2/3,       # labor factor
                 c_f = 15,      # fixed cost
                 c_e = 50,      # entry cost
                 μ_s = 1.2,     # loc of initial productivity draw
                 ρ = 0.9,       # productivity autocorrelation
                 σ_ϵ = 0.2,     # shock coefficient
                 τ = 0.0,       # firing cost coefficient
                 T = 5,         # length of cohort tracking
                 Ns = 20,       # number of productivity states
                 Nn = 200,      # number of previous labor states
                 VFI_iter_count = 100,
                 price_loop_count = 1,
                 distribution_count = 25,
                 distribution_solver_method = "transition_matrix",
                 labor_adj_cost_form = "linear",
                 κ = 0.1,
                 display_plots=True,
                 verbose=False
                 ):
        self.β, self.B, self.α, self.c_f, self.c_e, self.μ_s, self.ρ, self.σ_ϵ, self.Ns = β, B, α, c_f, c_e, μ_s, ρ, σ_ϵ, Ns
        self.τ, self.T, self.Nn = τ, T, Nn
        self.VFI_iter_count, self.price_loop_count = VFI_iter_count, price_loop_count
        self.distribution_count, self.distribution_solver_method = distribution_count, distribution_solver_method
        self.labor_adj_cost_form, self.κ = labor_adj_cost_form, κ
        self.display_plots = display_plots
        self.verbose = verbose
        self.setup_parameters()
        self.setup_grids()


    def setup_parameters(self):
        # incumbent firm solution
        self.tol = 1e-6
        self.maxit = 2_000

    
    def setup_grids(self):
        self.setup_s_grid()
        self.setup_n_grid()
        self.setup_labor_adj_grid()
        self.setup_production_grid()


    def setup_s_grid(self):
        # discretely approximate the continuous AR(1) process 
        self.mc = qe.markov.approximation.rouwenhorst(n=self.Ns, rho=self.ρ, sigma=self.σ_ε, mu=self.μ_s*(1-self.ρ))

        # transition matrix and states
        self.F = self.mc.P
        self.s_grid = np.exp(self.mc.state_values)
        
        # initial productivity distribution for entrant firm
        self.G = np.array(self.mc.stationary_distributions)
        self.G = np.squeeze(np.asarray(self.G))

    
    def setup_n_grid(self):
        n_max = self.optimal_n_without_frictions(self.s_grid[-1],1)
        vn = np.concatenate((np.arange(100), 10 ** np.linspace(2, np.log10(n_max), self.Nn - 100)))
        vn = np.unique(np.round(vn)).astype(int)
        vn = vn.reshape(-1, 1)
        self.Nn = len(vn)
        self.n_grid = vn.ravel()

    
    def setup_labor_adj_grid(self):
        adj = np.ones((self.Nn,self.Nn))
        for i in range(self.Nn):
            for i_prime in range(self.Nn):
                adj[i,i_prime] = self.labor_adj_cost(self.n_grid[i],self.n_grid[i_prime])
        self.labor_adj_grid = adj
    

    def setup_production_grid(self):
        production_grid = np.zeros((self.Nn,self.Ns))
        for jn, n in enumerate(self.n_grid):
            for js, s in enumerate(self.s_grid):
                production_grid[jn,js] = self.production(n,s)
        self.production_grid = production_grid


    def interpol(self,x,y,x1):
        y1 = np.interp(x1,x,y)
        return y1
    

    ####################
    # Functional Forms #
    ####################
    

    def hh_utility(self,C,N):
        u = np.log(C) - (1/self.B)*N
        return u


    def production(self,n,s):
        return s * n ** self.α


    def optimal_n_without_frictions(self,s,p):
        optimal_n = (self.α * p * s)**(1/(1-self.α))
        return optimal_n
    

    def labor_adj_cost(self, previous_n, n):
        if self.labor_adj_cost_form == "linear":
            cost = self.τ * np.maximum(0, previous_n - n)
        elif self.labor_adj_cost_form == "quadratic":
            cost = (self.κ / 2) * ((n - previous_n) ** 2)
        return cost


    def incumbent_firm(self,price,VF_old=None):
        """
        Value function iteration for the incumbent firm problem.
        """
        # setup
        Nn, Ns = self.Nn, self.Ns
        if VF_old is None:
            VF_old = -np.ones((Nn,Ns))  #initial guess

        VF = -np.ones((Nn,Ns))

        scrap_value = -np.repeat(self.labor_adj_grid[:, 0][:, np.newaxis], Ns, axis=1)  # nNprime x Ns matrix
        profit_mat = np.zeros((Nn, Ns, Nn))  # 1st is previous_n 2nd is n
        for ipN in range(Nn):
            for iN in range(Nn):
                for iS in range(Ns):
                    profit_mat[ipN, iS, iN] = price * self.production_grid[iN, iS] - self.n_grid[iN] - price * self.c_f - self.labor_adj_grid[ipN, iN]

        # VFI
        for it in range(self.maxit):
            expected_value = VF_old @ (self.F.T)
            continue_value = self.β * np.maximum(expected_value, scrap_value) # n x S matrix

            for iS in range(Ns):
                for iN in range(Nn):
                    VF[iN,iS] = np.max(profit_mat[iN,iS,:] + continue_value[:,iS])

            sup = np.max(np.abs(VF_old - VF))  # check tolerance
            VF_old[:] = VF[:]
            if sup < self.tol: 
                if self.verbose: print(f"     VFI iteration: {it}. Tol. achieved: {sup:.2E}")
                break
            if (it+1 == self.maxit) and self.verbose: print(f"Max iterations achieved. VF did not converge: {sup:.2E}")
            if (it % self.VFI_iter_count == 0) and self.verbose: print(f"     VFI iteration: {it}. Tol: {sup:.2E}")

        # recover policy functions
        ExpV = VF @ (self.F.T)
        ConV = self.β * np.maximum(ExpV, scrap_value)  # nPrime x Z matrix
        
        exit_indicator = (scrap_value > ExpV).astype(int)  # exit policy (1 == exit)
        continue_indicator = 1 - exit_indicator
        
        npi = np.zeros((Nn, Ns), dtype=int)  # recover index
        profit = np.zeros((Nn, Ns))  # optimal profits
        tax_r = np.zeros((Nn, Ns))  # taxes
        inaction_indicator = np.zeros((Nn,Ns))

        for iS in range(Ns):
            for iN in range(Nn):
                npi[iN, iS] = np.argmax(profit_mat[iN, iS, :] + ConV[:, iS])
                inaction_indicator[iN,iS] = (npi[iN,iS] == iN)
                profit[iN, iS] = profit_mat[iN, iS, npi[iN, iS]]
                tax_r[iN, iS] = self.labor_adj_grid[iN, npi[iN, iS]] * continue_indicator[iN,iS] + exit_indicator[iN, iS] * self.labor_adj_grid[iN, 1]

                # n_grid_matrix = np.tile(self.n_grid.reshape(-1, 1), (1, self.Ns))
        n_pol = self.n_grid[npi]  # employment values
        inaction_values = np.multiply(n_pol, inaction_indicator)

        VF_data = {
            "VF": VF, 
            "continue_indicator": continue_indicator, 
            "inaction_indicator": inaction_indicator, 
            "inaction_values": inaction_values, 
            "n_grid": self.n_grid, 
            "s_grid": self.s_grid, 
            "npi": npi, 
            "n_pol": n_pol
            }

        return VF_data


    ##############
    # Price Loop #
    ##############


    def entrant_firm(self,VF_incumbent):
        VF_entrant = VF_incumbent[0,:] @ self.G
        return VF_entrant


    def find_equilibrium_price(self):
        """
        Finds the equilibrium price that clears markets. 
        
        The function follows steps 1-3 in algorithm using the bisection method. It guesses a price, solves incumbent firm vf 
        then checks whether free entry condition is satisfied. If not, updates the price and tries again. The free entry condition 
        is where the firm value of the entrant (VF_entrant) equals the cost of entry (ce) and hence the difference between the two is zero. 
        """
        pmin, pmax = 0, 5
        for it_p in range(self.maxit):
            price = (pmin+pmax)/2
            if it_p == 0:
                VF = self.incumbent_firm(price)["VF"]
            else:
                VF = self.incumbent_firm(price,VF_old=VF)["VF"]

            VF_entrant = self.entrant_firm(VF)
            
            diff = VF_entrant-(price*self.c_e)
            if self.verbose:
                if it_p % self.price_loop_count == 0:
                    print(f"Price loop iteration {it_p}: diff = {diff}, price = {price}, VF_entrant = {VF_entrant}")
            if np.abs(diff) < self.tol:
                break
            
            if VF_entrant < price*self.c_e :
                pmin=price 
            else:
                pmax=price

        return price


    def alt_price_finder(self):
        def entry(p_guess):
            VF = self.incumbent_firm(p_guess)["VF"]
            excess_entry = np.dot(VF[0,:], self.G) - p_guess * self.c_e
            print("Excess entry: ", excess_entry, "price: ", p_guess)
            return excess_entry

        p0 = 0.1; p1 = 4.0 # guess: lower and upper bound. Might have to change for diff. parameters
        c_e = self.c_e
        p = sp.optimize.brentq(entry, p0, p1, xtol = 0.01)
        VF = self.incumbent_firm(p)

        return p
    

    ######################
    # Solve distribution #
    ######################


    def solve_dist_with_iteration(self, npi, continue_indicator, n_pol, price):
        inv_dist = np.ones((self.Nn, self.Ns))
        inv_dist[0,:] = self.G
        dsn_next = np.ones((self.Nn, self.Ns))

        for iter in range(self.maxit):
            dsn_next[:] = 0.0

            for jn in range(self.Nn):
                for js in range(self.Ns):
                    dsn_next[npi[jn,js],:] += inv_dist[jn,js]*(self.F)[js,:]*continue_indicator[jn,js]
            dsn_next[0,:] += self.G

            sup = np.max(np.abs(dsn_next - inv_dist))
            inv_dist[:] = dsn_next

            if sup < 1e-8:
                if self.verbose:
                    print(f"Iter: {iter}. Tol. achieved: {sup:.2E}")
                    break
            if (iter == self.maxit) & self.verbose: print(f"Max iterations achieved. Inv. dist did not converge: {sup:.2E}")
            if (iter%self.distribution_count==0) & self.verbose: print(f"Distribution iter: {iter}. Tol: {sup:.2E}")

        gamma_at_1 = inv_dist
        # print(f"gamma_at_1 shape: {np.shape(gamma_at_1)}")

        optimal_production = self.production(n_pol, self.s_grid)
        Y = self.B / price
        m = Y / (np.dot(gamma_at_1.flatten(), optimal_production.flatten() - self.c_f) - self.c_e)
        gamma = m * gamma_at_1
        # print(f"gamma shape: {np.shape(gamma_at_1)}")
        # gamma = np.reshape(self.Nn,self.Ns)
        mass_all = np.sum(gamma)

        invariant_distribution_data = {
            "gamma_at_1": gamma_at_1,
            "gamma": gamma,
            "m": m,
            "M": mass_all,
            "Y": Y,
            }

        return invariant_distribution_data
    

    def build_transition_matrix(self,npi,continue_indicator):
        transition = np.zeros((self.Nn,self.Ns,self.Nn,self.Ns))

        for js in range(self.Ns):
            for jn_1 in range(self.Nn):
                for jsp in range(self.Ns):
                    if continue_indicator[jn_1,js] == 1:
                        transition[jn_1,js,npi[jn_1,js],jsp] += self.F[js,jsp]

        transition = np.reshape(transition, newshape=(self.Nn*self.Ns, self.Nn*self.Ns))

        return transition


    def solve_dist_with_transition_matrix(self,npi,continue_indicator,n_pol,price,transition_matrix = None):
        # Build the transition matrix
        if transition_matrix is None:
            transition = self.build_transition_matrix(npi, continue_indicator)
        else:
            transition = transition_matrix

        # Create the identity matrix
        speye = np.eye(self.Ns * self.Nn)
        sparse_vector = np.vstack([np.reshape(self.G, (self.Ns, 1)), np.zeros((self.Ns * (self.Nn - 1), 1))])
        gamma_at_1 = np.linalg.solve(speye - transition.T, sparse_vector)
    
        my = (self.production(n_pol, self.s_grid))
        Y = self.B / price
        m = (Y / (np.dot(gamma_at_1.T, my.flatten() - self.c_f) - self.c_e))[0]
        gamma = m * gamma_at_1
        mSSD = np.reshape(gamma, newshape=(self.Nn,self.Ns))
        mass_all = np.sum(mSSD)
        
        invariant_distribution_data = {
            "gamma_at_1": gamma_at_1,
            "gamma": gamma,
            "m": m,
            "M": mass_all,
            "Y": Y,
            "transition_matrix": transition
            }

        return invariant_distribution_data
    

    def track_cohort(self, n, transition_matrix):
        exit_rate = np.zeros(shape=n)
        relative_size = np.zeros(shape=n+1)
        current_dist = np.zeros((self.Nn,self.Ns))
        continue_indicator = self.continue_indicator.flatten()
        n_pol = self.n_pol.flatten()
        for age in range(n+1):
            if age == 0:
                current_dist[0,:] = self.G
                current_dist = current_dist.flatten()
                current_size = self.m
                current_gamma = current_dist * current_size
            else:
                current_gamma = transition_matrix.T @ (np.multiply(current_gamma,continue_indicator))
                current_size = np.sum(current_gamma)
                current_dist = current_gamma / current_size
            
            current_average_size = np.dot(n_pol, current_dist)
            relative_size[age] = current_average_size / self.average_firm_size
            current_continue_rate = np.dot(current_dist,continue_indicator)
            if age > 0:
                exit_rate[age-1] = 1 - current_continue_rate

            print(f"Age: {age}")
            if age > 0:
                print(f"{'Exit rate ':.<40}{exit_rate[age-1]:0.4f}")
            print(f"{'Relative firm size ':.<40}{relative_size[age]:0.4f}")
        if self.display_plots == True:
            fig, axes = plt.subplots(1,2,figsize=(10,5))
            axes[0].plot(range(1,n+1), exit_rate)
            axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=None))
            axes[0].set_title(label="Exit Rate")
            axes[1].plot(range(n+1), relative_size)
            axes[1].set_title(label="Relative Size")
            plt.show()

    
    def report_misallocation(self):
        MPL = np.zeros((self.Nn,self.Ns))
        mpl = np.zeros((self.Nn,self.Ns))
        for jn in range(self.Nn):
            for js in range(self.Ns):
                if self.n_pol[jn,js] > 0:
                    MPL[jn,js] = self.price_ss * self.α * self.s_grid[js] * (self.n_pol[jn,js] ** (self.α - 1))
                    #if MPL[jn,js] > 0:
                    mpl[jn,js] = np.log(MPL[jn,js])
        
        mpl_std = np.std((mpl * self.stationary_pdf)[self.stationary_pdf != 0])

        avg_abs_mpl = np.sum(np.abs(mpl) * self.stationary_pdf)
        upper_region = np.sum((mpl > avg_abs_mpl) * self.stationary_pdf)
        lower_region = np.sum((mpl < -avg_abs_mpl) * self.stationary_pdf)

        total_region = lower_region + upper_region

        # report findings
        report_dict = {
            "Standard Deviation in mpl": mpl_std,
            "Average |mpl|": avg_abs_mpl,
            "Fraction of mpl > avg.|mpl|": upper_region,
            "Fraction of mpl < avg.|mpl|": lower_region,
            "Fraction of firms outside range": total_region
            }
        
        print("\n-----------------------------------------------------------")
        for string, var in report_dict.items():
            print(f"{string:.<40} {var:0.6f}")
        print("-----------------------------------------------------------")

        return report_dict


    #################
    # Main function #
    #################


    def solve_model(self, price=None):
        print("Solving model...")
        # find price
        if price is not None:
            self.price_ss = price
        else:
            self.price_ss = self.find_equilibrium_price()

        # find value fucntions at equilibrium price
        VF_data = self.incumbent_firm(self.price_ss)
        self.VF_incumbent = VF_data["VF"]
        self.VF_entrant = self.entrant_firm(self.VF_incumbent)
        self.npi = VF_data["npi"]
        self.continue_indicator=VF_data["continue_indicator"]
        self.n_pol=VF_data["n_pol"]

        # find distribution
        if self.distribution_solver_method == "iteration":
            distribution_data = self.solve_dist_with_iteration(npi = self.npi,
                                                                           continue_indicator=self.continue_indicator,
                                                                           n_pol=self.n_pol,
                                                                           price = self.price_ss)
            self.transition_matrix = self.build_transition_matrix(self.npi,self.continue_indicator)
        elif self.distribution_solver_method == "transition_matrix":
            distribution_data = self.solve_dist_with_transition_matrix(npi = self.npi,
                                                                           continue_indicator=self.continue_indicator,
                                                                           n_pol=self.n_pol,
                                                                           price = self.price_ss)
            self.transition_matrix = distribution_data["transition_matrix"]
        else:
            print("Distribution solver method not recognized. Using transition matrix.")
            distribution_data = self.solve_dist_with_transition_matrix(npi = self.npi,
                                                                           continue_indicator=self.continue_indicator,
                                                                           n_pol=self.n_pol,
                                                                           price = self.price_ss)
            self.transition_matrix = distribution_data["transition_matrix"]
        self.gamma = np.reshape(distribution_data["gamma"], (self.Nn,self.Ns))
        self.m = distribution_data["m"]
        self.M = distribution_data["M"]
        self.Y = distribution_data["Y"]
        self.stationary_pdf = self.gamma / self.M
        self.stationary_cdf = np.cumsum(self.stationary_pdf)

        self.employment_distribution = self.n_pol * self.gamma

        self.pdf_emp = self.employment_distribution / np.sum(self.employment_distribution)
        self.cdf_emp = np.cumsum(self.pdf_emp)

        # calculate statistics
        self.total_employment = np.tensordot(self.n_pol, self.gamma)
        self.average_firm_size = self.total_employment / self.M
        self.average_entrant_size = np.dot(self.n_pol[0,:], self.G)
        self.relative_size_of_new_entrants = self.average_entrant_size / self.average_firm_size
        self.exit_rate = self.m / self.M

        # report findings
        self.report_dict = {
            "Steady State Equilibrium Price": self.price_ss,
            "Entry Mass": self.m,
            "Mass of all firms": self.M,
            "Firm entry/exit rate": self.exit_rate,
            "Average Firm Size": self.average_firm_size,
            "Relative Size of New Entrants": self.relative_size_of_new_entrants,
            "Aggregate Output": self.Y,
            "Aggregate Employment": self.total_employment
            }
        
        print("\n-----------------------------------------------------------")
        for string, var in self.report_dict.items():
            print(f"{string:.<40} {var:0.4f}")
        print("-----------------------------------------------------------")

        self.track_cohort(5, self.transition_matrix)

        self.report_misallocation()

        return self.report_dict

def plot_inaction_regions(tau_vals = None, price = None, verbose=False):
    fig, ax = plt.subplots()
    plt.ylim(0,1000)
    plt.ylabel(r"$n_{-1}$")
    plt.xlim(0,15)
    plt.xlabel(r"$s$")
    bench_model = HR93(τ=0, verbose=verbose)
    if price is not None:
        bench_price = price
    else:
        bench_price = bench_model.find_equilibrium_price()
    bench_VF_data = bench_model.incumbent_firm(price=bench_price)
    s_grid = bench_VF_data["s_grid"]
    bench_values = [bench_model.optimal_n_without_frictions(s, p=bench_price) for s in s_grid]
    ax.plot(s_grid, bench_values, 'k-', label="Benchmark")

    if tau_vals is not None:
        cmap = plt.get_cmap('Blues')
        num_plots = len(tau_vals)
        colors = [cmap(1 - 0.8*i / num_plots) for i in range(len(tau_vals))]
        for i, tau in enumerate(tau_vals):
            model = HR93(τ=tau, verbose=verbose)
            #price = model.find_equilibrium_price()
            inaction_values = model.incumbent_firm(price=bench_price)["inaction_values"]
            def min_exclude_zeros(column):
                filtered_column = column[column > 0]
                return np.min(filtered_column) if filtered_column.size > 0 else np.nan
            
            inaction_lower = np.apply_along_axis(min_exclude_zeros, axis=0, arr=inaction_values)
            inaction_upper = np.max(inaction_values, axis=0)
            plt.fill_between(s_grid, inaction_lower, inaction_upper, color=colors[i], alpha=0.5, label=fr'$\tau = {tau}$')

    plt.legend()
    plt.title(label="Inaction Regions")
    plt.show()

# model = HR93(τ=0.1, labor_adj_cost_form="linear")
# results = model.solve_model()

# plot_inaction_regions(tau_vals=[0.1,0.2,0.3], price=.9699)