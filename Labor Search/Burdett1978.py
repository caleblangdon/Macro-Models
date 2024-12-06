import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

class Burdett1978:
    """
    Class object of the Burdett (1978) model.
    """

    ##############
    # Initialize #
    ##############

    def __init__(self,
                 δ = .03,                   # exogeneous firing rate
                 s = .5,                    # search effort of employed
                 λ = .4,                    # job finding rate of unemployed
                 β = .995,                  # discrete discount factor
                 r = None,                  # continuous discount rate
                 b = .4*1,                  # unemployment benefit / home production
                 Nw = 100,                  # number of points in w grid
                 mean = 1,                  # mean of lognormal dist
                 variance = .1,             # variance of lognormal dist
                 params_for_normal = False, # says whether given mean and var are for Z instead of w
                 low_lim = 0,               # lower truncation limit of log normal
                 up_lim = 3.5,              # upper truncation limit of log normal
                 max_iter = 10_000,         # max number of iterations
                 tol = 1e-5,                # tolerance
                 automatically_discretize=True,
                 truncate = True,
                 verbose = True):
        self.δ, self.s, self.λ, self.β, self.Nw = δ, s, λ, β, Nw # store params as class objects
        self.r = -np.log(self.β) if r is None else r  # calculate r if not provided
        #self.b = b
        self.b = .4*mean if not params_for_normal else .4*np.exp(mean + .5*variance)
        self.params_are_for_underlying_normal = params_for_normal
        self.lower_limit, self.upper_limit = low_lim, up_lim
        self.tol, self.max_iter, self.truncate, self.verbose = tol, max_iter, truncate, verbose

        self.f, self.F, self.underlying_dist = self.setup_F(mean, variance, low_lim, up_lim)
        self.w_R = self.solve_w_R()
        self.g, self.G = self.setup_G(self.w_R)
        if automatically_discretize:
            self.w_grid, self.f_grid, self.F_grid, self.g_grid, self.G_grid = self.discretize()
        self.ss_u = self.δ / (self.δ + self.λ * (1-self.F(self.w_R)))
        if self.verbose:
            print(f"Model initialized with reservation wage {self.w_R:.4f} and steady state unemployment {self.ss_u:.4f}")
    
    ###################################
    # Solve Reservation Wage and G(w) #
    ###################################

    def setup_F(self, mean, var, lower_limit, upper_limit):
        std_dev = np.sqrt(var)
        # find mu, sigma of underlying normal
        if self.params_are_for_underlying_normal:
            mu = mean
            sigma_squared = var
        else:
            mu = np.log((mean**2)/(np.sqrt(var + (mean**2))))
            sigma_squared = np.log(1+(var/(mean**2)))
        sigma = np.sqrt(sigma_squared)

        a = -np.inf
        b = (np.log(self.upper_limit)-mu)/sigma if self.truncate else np.inf
        underlying_dist = sp.stats.truncnorm(a=a, b=b, loc=mu, scale=sigma)
        
        # setup distribution using calculated params of underlying normal dist
        # F_dist = sp.stats.lognorm(s=sigma, scale=np.exp(mu))
        # print(f"{F_dist.mean() = }")
        # print(f"{F_dist.var() = }")

        # truncation_factor = F_dist.cdf(upper_limit) - F_dist.cdf(lower_limit)
        def wage_pdf(w):
            # if w > 0:
            #     result = underlying_dist.pdf(np.log(w)) / w
            # else:
            #     result = 0

            w = np.asarray(w)
            result = np.zeros_like(w)
            result[w>0] = self.underlying_dist.pdf(np.log(w[w>0]))/w[w>0]
            return result
        
        def wage_cdf(w):
            # if w > 0:
            #     result = underlying_dist.cdf(np.log(w))
            # else:
            #     result = 0
            w = np.asarray(w)
            result = np.zeros_like(w)
            result[w>0] = self.underlying_dist.cdf(np.log(w[w>0]))
            return result
        
        return wage_pdf, wage_cdf, underlying_dist

    def solve_w_R(self):
        def integrand(w):
            return (1 - self.F(w)) / (self.r + self.δ + self.s * self.λ * (1 - self.F(w)))
        def equation(w_R):
            integral, _ = sp.integrate.quad(integrand, w_R, self.upper_limit)
            return w_R - (self.b + self.λ * (1 - self.s) * integral)
        
        result = sp.optimize.root_scalar(equation, bracket=[self.lower_limit, self.upper_limit])
        return result.root

    def setup_G(self, w_R):
        def G(w):
            if w < self.w_R:
                result = 0
            else:
                result = (self.δ * (self.F(w) - self.F(w_R))) / ((1-self.F(w_R)) * (self.s * self.λ * (1-self.F(w)) + self.δ))
            return result
        def g(w):
            if w < self.w_R:
                result = 0
            else:
                num = self.f(w) * self.δ * (self.δ + self.s * self.λ * (1-self.F(w)) + self.s*self.λ*(self.F(w)-self.F(self.w_R)))
                denom = (1-self.F(self.w_R)) * ((self.δ + self.s * self.λ * (1-self.F(w)))**2)
                result = num / denom
            return result
        return g, G
    
    def F_percentile(self, p):
        def equation(w):
            return self.F(w) - p
        result = sp.optimize.root_scalar(equation, bracket=[self.lower_limit, self.upper_limit])
        return result.root
    
    def discretize(self):
        w_01 = self.F_percentile(.01)
        w_99 = self.F_percentile(.99)
        w_grid = np.linspace(w_01, w_99, self.Nw)

        f_grid = [self.f(w) for w in w_grid]
        f_grid = f_grid / np.sum(f_grid)
        F_grid = np.cumsum(f_grid)

        g_grid = [self.g(w) for w in w_grid]
        g_grid = g_grid / np.sum(g_grid)
        G_grid = np.cumsum(g_grid)

        return w_grid, f_grid, F_grid, g_grid, G_grid
    
    def avg_accepted_wage(self):
        def integrand(w):
            return w * self.g(w)
        w_bar =sp.integrate. quad(integrand, self.w_R, self.upper_limit)[0]
        return w_bar

    def var_wages(self, w_bar=None):
        w_bar = self.avg_accepted_wage() if w_bar is None else w_bar
        def integrand(w):
            return ((w - w_bar)**2) * self.g(w)
        var = sp.integrate.quad(integrand, self.w_R, self.upper_limit)[0]
        return var

    def var_log_wages(self, w_bar=None):
        w_bar = self.avg_accepted_wage() if w_bar is None else w_bar
        def integrand(w):
            return ((np.log(w) - np.log(w_bar))**2) * self.g(w)
        var = sp.integrate.quad(integrand, self.w_R, self.upper_limit)[0]
        return var

    def G_percentile(self, p):
        def equation(w):
            return self.G(w) - p
        result = sp.optimize.root_scalar(equation, bracket=[self.w_R, self.upper_limit])
        return result.root

    def chi(self):
        def integrand(w):
            return (1-self.F(w))*self.g(w)
        integral = sp.integrate.quad(integrand, self.w_R, self.upper_limit)[0]
        result = self.s * self.λ * integral
        return result
    
    def simulate_to_ss(self, N=100):
        # create vector of workers
        theoretical_u = self.δ / (self.δ + self.λ * (1-self.F(self.w_R)))
        avg_wage = self.avg_accepted_wage()
        worker_status = np.zeros(N)
        worker_wages = np.zeros(N)
        for t in range(self.max_iter):
            for i in range(N):
                if worker_status[i] == 0: # unemployed
                    if np.random.rand() < self.λ:
                        wage_offer = np.exp(self.underlying_dist.rvs())
                        if wage_offer >= self.w_R:
                            worker_status[i] = 1
                            worker_wages[i] = wage_offer

                elif worker_status[i] == 1: # employed
                    if np.random.rand() < self.δ: #exogeneous separation
                        worker_status[i] = 0
                        worker_wages[i] = 0
                    elif np.random.rand() < self.s * self.λ:
                        wage_offer = np.exp(self.underlying_dist.rvs())
                        if wage_offer > worker_wages[i]:
                            worker_wages[i] = wage_offer
            
            u = np.sum(worker_status == 0)
            u_rate = u/N
            employed_workers_wages = worker_wages[worker_status == 1]

            if t % 25 == 0:
                print(f"Iteration {t}: {u_rate:.2%} unemployment rate")

            if t > 50 and np.isclose(avg_wage, np.mean(employed_workers_wages), rtol=1e-2) and np.isclose(theoretical_u, u_rate, rtol=1e-2):
                print(f"Converged to SS after {t} iterations with unemployment rate {u_rate:.2%}.")
                break

            if t == self.max_iter-1:
                print(f"Maximum iteration ({self.max_iter}) reached.")
            
        return worker_status, worker_wages, employed_workers_wages
    
    def simulate_history(self, worker_status_ss, worker_wages_ss, Y=27):
        N = len(worker_status_ss)
        T = Y*12
        worker_status_hist = np.zeros(shape=(N,T+1))
        worker_wages_hist = np.zeros(shape=(N,T+1))
        treatment_status_hist = np.zeros(shape=(N))
        worker_status_hist[:,0] = worker_status_ss
        worker_wages_hist[:,0] = worker_wages_ss

        for t in range(1,T+1):
            for i in range(N):
                if worker_status_hist[i,t-1] == 0: # unemployed
                    if np.random.rand() < self.λ:
                        wage_offer = np.exp(self.underlying_dist.rvs())
                        if wage_offer >= self.w_R:
                            worker_status_hist[i,t] = 1
                            worker_wages_hist[i,t] = wage_offer

                elif worker_status_hist[i,t-1] == 1: # employed
                    if np.random.rand() < self.δ: #exogeneous separation
                        worker_status_hist[i,t] = 0
                        worker_wages_hist[i,t] = 0
                        if t in range((6*12)+1, (7*12)+1):
                            treatment_status_hist[i] = 1
                    else: # doesn't separate
                        worker_status_hist[i,t] = 1
                        worker_wages_hist[i,t] = worker_wages_hist[i,t-1]
                        if np.random.rand() <self.s*self.λ:
                            wage_offer = np.exp(self.underlying_dist.rvs())
                            if wage_offer > worker_wages_hist[i,t-1]:
                                worker_wages_hist[i,t] = wage_offer
        
        monthly_data = pd.DataFrame({
            'worker_id': np.repeat(np.arange(N), T),
            'month': np.tile(np.arange(1,T+1), N),
            'wage': worker_wages_hist[:,1:].reshape(N*T),
            'earnings' : worker_wages_hist[:,1:].reshape(N*T) * worker_status_hist[:,1:].reshape(N*T),
            'employment_status': worker_status_hist[:,1:].reshape(N*T),
            "treatment_status": np.repeat(treatment_status_hist, T)
        })

        # Convert month to year
        monthly_data['year'] = ((monthly_data['month']-1) // 12)+1

        # Group by worker_id and year, then calculate mean earnings, mean wages, and mean employment
        yearly_data = monthly_data.groupby(['worker_id', 'year']).agg(
            mean_earnings=('wage', 'mean'),
            mean_wages=('wage', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else np.nan),  # Mean wages only for employed months
            mean_employment=('employment_status', 'mean'),
            treatment_status=("treatment_status", 'mean')
        ).reset_index()

        # Fill missing values in mean_wages with 0
        yearly_data['mean_wages'].fillna(0, inplace=True)
        
        # Set multi-level index
        # yearly_data.set_index(['worker_id', 'year'], inplace=True)

        for k in range(-6, Y-6):
            col_name = f"D_{k}"
            yearly_data[col_name] = 0  # Initialize the column with zeros
            yearly_data.loc[(yearly_data['year'] - k == 7) & (yearly_data['treatment_status'] == 1), col_name] = 1

        return monthly_data, yearly_data
    
    def solve_value_functions(self, vectorize=True):
        wri = np.argmin(np.abs(self.w_R - self.w_grid))
        wr = self.w_grid[wri]
        U0 = self.b / (1-self.β)
        V0 = self.w_grid.copy() / (1-self.β)
        U1 = self.b / (1-self.β)
        V1 = self.w_grid.copy() / (1-self.β)

        print(f"{wr = :.4f}, {wri = }")

        for it in range(self.max_iter):
            U1 = self.b + self.β * ((1-self.λ)*U0 + self.λ * np.dot(np.maximum(U0, V0), self.f_grid))
            
            # for-loop
            # for iw, w in enumerate(self.w_grid):
            if not vectorize:
                for iw, w in enumerate(self.w_grid):
                    V1[iw] = w + self.β * (self.δ * U0 + (1-self.δ)*((1-self.s*self.λ)*V0[iw] 
                                                        + self.s*self.λ*np.dot(np.maximum(V0[iw], V0), self.f_grid)))

            else:
                    V1 = self.w_grid + self.β * (self.δ * U0 + (1 - self.δ) *
                                                        ((1 - self.s * self.λ) * V0 + 
                                                        self.s * self.λ * np.dot(np.maximum(V0[:, None], V0), self.f_grid)))
            
            U_error = np.abs(U1-U0)
            V_error = np.abs(V1-V0)
            error = max(U_error,np.max(V_error))
            U0, V0 = U1, V1

            if (it+1) % 100 == 0 and self.verbose:
                print(f"Iteration {it+1}. Largest error = {error:.6f}")
            if error < self.tol and self.verbose:
                print(f"Converged after {it+1} iterations.")
                break
            if it == self.max_iter-1 and self.verbose:
                print(f"Maximum iterations reached.")
                
        print(f"{U1 = :.4f}, {V1[wri] = :.4f}")
        return U1, V1
    
    def calculate_pv_avg_of_earnings(self, earnings_data):
        # Identify workers who are unemployed at the beginning of the data
        initial_unemployed_workers = earnings_data[earnings_data['month'] == 1]
        initial_unemployed_workers = initial_unemployed_workers[initial_unemployed_workers['employment_status'] == 0]['worker_id'].unique()

        # Filter earnings data to include only these workers
        earnings_data = earnings_data[earnings_data['worker_id'].isin(initial_unemployed_workers)].copy()

        # Group by worker_id and month, then sum the discounted earnings
        avg_earnings = earnings_data['earnings'].mean()
        pv_of_avg_earning = avg_earnings / (1 - self.β)
        if self.verbose:
            print(f"Present value of earnings, averaged across initially unemployed workers: {np.average(pv_of_avg_earning)}")

        return pv_of_avg_earning
    
    def display_distributions(self):
        # Define a range of wage values to evaluate the PDF and CDF
        w_values = np.linspace(0-.5, 3.5+.5, 1000)

        # Evaluate the PDF's and CDF's over this range
        F_values = [self.F(w) for w in w_values]
        f_values = [self.f(w) for w in w_values]
        G_values = [self.G(w) for w in w_values]
        g_values = [self.g(w) for w in w_values]

        # plot the PDF's
        plt.plot(w_values, f_values, label='f')
        plt.plot(w_values, g_values, label='g')
        plt.xlabel('Wage')
        plt.ylabel('Probability')
        plt.title('PDF')
        plt.legend()
        plt.grid()
        plt.show()

        # plot the CDF's
        plt.plot(w_values, F_values, label='F')
        plt.plot(w_values, G_values, label='G')
        plt.xlabel('Wage')
        plt.ylabel('Cumulative Probability')
        plt.title('CDF')
        plt.legend()
        plt.grid()
        plt.show()
    
    def report_moments(self):
        avg_wage = self.avg_accepted_wage()
        variance_log_wages = self.var_log_wages()
        log_50_10_diff = np.log(self.G_percentile(0.5)) - np.log(self.G_percentile(0.1))
        j2j_prob = self.chi()

        print(f"Reservation wage (w_R): {self.w_R:.4f}")
        print(f"Average wage: {avg_wage:.4f}")
        print(f"Variance of log wages: {variance_log_wages:.4f}")
        print(f"Log 50-10 differential: {log_50_10_diff:.4f}")
        print(f"Average job-to-job transition probability: {j2j_prob:.4f}")

    def compare_distributions(self, employed_workers_wages_ss):
        w_vals = np.linspace(self.lower_limit, self.upper_limit, self.Nw)
        G_simulated = [np.mean(employed_workers_wages_ss <= w) for w in w_vals]
        G_theoretical = [self.G(w) if w >= self.w_R else 0 for w in w_vals]

        plt.plot(w_vals, G_theoretical, label='Theoretical G(w)', alpha=0.75)
        plt.plot(w_vals, G_simulated, label='Simulated G(w)', linestyle='--', alpha=0.75)
        plt.xlabel('Wage')
        plt.ylabel('Cumulative Distribution Function')
        plt.legend()
        plt.title('Comparison of Simulated and Theoretical Wage Distributions')
        plt.grid()
        plt.show()

    def job_loss_scar(self, yearly_data):
        dummy_names = [f"D_{k}" for k in range(-6, 27-6)]
        results = []

        fig, axes = plt.subplots(1,3, figsize=(18, 5))

        for i, dep_var in enumerate(['mean_wages', 'mean_earnings', 'mean_employment']):
            X = yearly_data[dummy_names]
            X = sm.add_constant(X)  # Adds a constant term to the exog matrix
            OLS_model = sm.OLS(endog=yearly_data[dep_var], exog=X)
            result = OLS_model.fit()
            results.append(result)
            coefs = result.params[dummy_names]
            axes[i].plot(range(-6, 21), coefs)
            axes[i].set_title(f'Dummy Coefficients for {dep_var}')
            axes[i].set_xlabel('Dummy Variable')
            axes[i].set_ylabel('Coefficient')
            axes[i].grid(True)
        plt.show()


# model = Burdett1978(verbose=True)
# model.display_distributions()
# model.report_moments()
# workers_status_ss, workers_wages_ss, employed_workers_wages_ss = model.simulate_to_ss(N=2_000)
# model.compare_distributions(employed_workers_wages_ss)
# monthly_data, yearly_data = model.simulate_history(workers_status_ss, workers_wages_ss)
# model.job_loss_scar(yearly_data)
# U, V = model.solve_value_functions(vectorize=True)
# pv_of_avg_earning = model.calculate_pv_avg_of_earnings(monthly_data)