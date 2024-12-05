# imports
import time
import numpy as np
import scipy as sp
import quantecon as qe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_theme(style='whitegrid')

class Hopenhayn1992:
    """
    Class object of the Hopenhayn (1992) model.
    """

    #################
    # 1. Initiation #
    #################

    def __init__(self,
                 β = 0.96,      # discount factor
                 B = 100,       # demand primitive
                 α = 2/3,       # labor factor
                 c_f = 15,      # fixed cost
                 c_e = 50,      # entry cost
                 μ_s = 1.2,     # loc of initial productivity draw
                 ρ = 0.9,       # productivity autocorrelation
                 σ_ϵ = 0.2,     # shock coefficient
                 Nz = 20,       # number of discrete states
                 display_plots=True,
                 ):
        self.β, self.B, self.α, self.c_f, self.c_e, self.μ_s, self.ρ, self.σ_ϵ, self.Nz = β, B, α, c_f, c_e, μ_s, ρ, σ_ϵ, Nz
        self.display_plots = display_plots
        self.price_ss = None
        self.setup_parameters()
        self.setup_grid()


    def setup_parameters(self):
        # incumbent firm solution
        self.tol = 1e-8
        self.maxit = 2_000


    def setup_grid(self):
        # discretely approximate the continuous AR(1) process 
        self.mc = qe.markov.approximation.rouwenhorst(n=self.Nz, rho=self.ρ, sigma=self.σ_ε, mu=self.μ_s*(1-self.ρ))

        # transition matrix and states
        self.F = self.mc.P
        self.prod_grid = np.exp(self.mc.state_values)

        # initial productivity distribution for entrant firm
        """ self.initial = qe.markov.approximation.rouwenhorst(n=self.Nz, rho=0, sigma=(self.σ_ε**2)/(1-self.ρ**2), mu=self.μ_s)
        self.G = np.array(self.initial.stationary_distributions)
        self.G = np.squeeze(np.asarray(self.G))
        self.grid_x = np.exp(self.initial.state_values) """

        self.G = np.array(self.mc.stationary_distributions)
        self.G = np.squeeze(np.asarray(self.G))


    def interpol(self,x,y,x1):
        y1 = np.interp(x1,x,y)
        return y1


    def static_profit_max(self, price):
        optimal_n = (self.α * price * self.prod_grid) ** (1 / (1 - self.α))
        firm_output = self.prod_grid * (optimal_n ** self.α)
        firm_profit = price*firm_output - optimal_n - price*self.c_f
        
        return firm_profit, firm_output, optimal_n


    def incumbent_firm(self, price):
        """
        Value function iteration for the incumbent firm problem.
        """ 
        VF_old = np.zeros(self.Nz)
        VF = np.zeros(self.Nz)
        
        firm_profit, firm_output, pol_n = self.static_profit_max(price)
    
        for it in range(self.maxit):
            
            VF = firm_profit + self.β * np.maximum(np.dot(self.F, VF_old),0)
            
            dist = np.abs(VF_old - VF).max()
        
            if dist < self.tol :
               break
           
            VF_old = np.copy(VF)

        continue_indicator = np.ones(self.Nz)*(np.dot(self.F, VF)>=0)
        
        idx = np.searchsorted(continue_indicator, 1) #index of self.pol_continue closest to one on the left
        exit_cutoff = self.prod_grid[idx]
        
        return VF, firm_profit, firm_output, pol_n, continue_indicator, exit_cutoff

    #################
    # 2. Price Loop #
    #################

    def find_equilibrium_price(self):
        """
        Finds the equilibrium price that clears markets. 
        
        The function follows steps 1-3 in algorithm using the bisection method. It guesses a price, solves incumbent firm vf 
        then checks whether free entry condition is satisfied. If not, updates the price and tries again. The free entry condition 
        is where the firm value of the entrant (VF_entrant) equals the cost of entry (ce) and hence the difference between the two is zero. 
        """
        
        pmin, pmax = 0, 100
        for it_p in range(self.maxit):
            price = (pmin+pmax)/2
            
            VF = self.incumbent_firm(price)[0]
        
            VF_entrant = np.dot(VF, self.G)
            
            diff = np.abs(VF_entrant-(price*self.c_e))
            
            if diff < self.tol:
                break
            
            if VF_entrant < price*self.c_e :
                pmin=price 
            else:
                pmax=price
        
        return price
    
    ########################
    # 2. Find Distribution #
    ########################

    def solve_invariant_distribution(self, m, continue_indicator):
        F_tilde = (self.F * continue_indicator.reshape(self.Nz, 1)).T
        I = np.eye(self.Nz)
         
        return m * ( np.dot( np.linalg.inv(I - F_tilde), self.G ) )

    ########################
    # 4. Further Inference #
    ########################
    def track_cohort(self, n):
        exit_rate = np.zeros(shape=n)
        relative_size = np.zeros(shape=n+1)
        for age in range(n+1):
            if age == 0:
                current_dist = self.G
                current_size = self.m_star
                current_gamma = current_dist * current_size
            else:
                current_gamma = self.F.T @ (np.multiply(current_gamma, self.continue_indicator))
                current_size = np.sum(current_gamma)
                current_dist = current_gamma / current_size
            
            current_average_size = np.dot(self.pol_n, current_dist)
            relative_size[age] = current_average_size / self.average_firm_size
            current_continue_rate = np.dot(current_dist,self.continue_indicator)
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
            axes[1].plot(range(n+1), relative_size)
            plt.show()
    

    def solve_model(self):
        """
        Finds the stationary equilibrium
        """  
        
        t0 = time.time()    #start the clock
        
        # Find the optimal price using bisection (algo steps 1-3)
        self.price_ss = self.find_equilibrium_price()
        
        # Use the equilibrium price to recover incumbent firm solution
        self.VF, self.firm_profit, self.firm_output, self.pol_n, self.continue_indicator, self.exit_cutoff = self.incumbent_firm(self.price_ss)
        
        # c. Invariant (productivity) distribution with endogenous exit. Here assume m=1 which 
        #will come in handy in the next step.
        self.gamma_at_1 = self.solve_invariant_distribution(1, self.continue_indicator)
        
        # d. Rather than iterating on market clearing condition to find the equilibrium mass of entrants (m_star)
        # we can compute it analytically (Edmond's notes ch. 3 pg. 25)
        self.Y = self.B / self.price_ss
        self.m_star = self.Y / (np.dot(self.gamma_at_1,self.firm_output - self.c_f) - self.c_e)
        
        # e. Rescale to get invariant (productivity) distribution (mass of plants)
        self.gamma = self.m_star * self.gamma_at_1
        self.total_mass = np.sum(self.gamma)
        
        # Invariant (productivity) distribution by percent
        self.pdf_stationary = self.gamma / self.total_mass
        self.cdf_stationary = np.cumsum(self.pdf_stationary)
        
        # f. calculate employment distributions
        self.distrib_emp = (self.pol_n * self.gamma)
        
        # invariant employment distribution by percent
        self.pdf_emp = self.distrib_emp / np.sum(self.distrib_emp)
        self.cdf_emp = np.cumsum(self.pdf_emp)
        
        # g. calculate statistics
        self.total_employment = np.dot(self.pol_n, self.gamma)
        self.average_firm_size = self.total_employment / self.total_mass
        self.average_entrant_size = np.dot(self.pol_n, self.G)
        self.relative_size_of_new_entrants = self.average_entrant_size / self.average_firm_size
        self.exit_rate = self.m_star / self.total_mass
        #self.exit_rate = 1-(np.sum(self.pi.T*self.distrib_stationary_0*self.pol_continue)/np.sum(self.distrib_stationary_0)) #alternative calculation

        self.report_dict = {
            "Steady State Equilibrium Price": self.price_ss,
            "Entry Mass": self.m_star,
            "Mass of all firms": self.total_mass,
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
        self.track_cohort(5)

        # h. plot
        
        if self.display_plots:
            plt.plot(self.prod_grid, self.VF)
            plt.axvline(self.exit_cutoff, color='tab:red', linestyle='--', alpha=0.7)
            plt.axhline(0, color='tab:green', linestyle='--', alpha=0.7)
            plt.title('Incumbant Firm Value Function')
            plt.legend(['Value Function', 'Exit Threshold='+str(self.exit_cutoff.round(2))])
            plt.xlabel('Productivity level')
            #plt.savefig('value_func_hopehayn.pdf')
            plt.show()
         
            plt.plot(self.prod_grid,self.pdf_stationary)
            plt.plot(self.prod_grid, self.pdf_emp)
            plt.title('Stationary PDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Density')
            plt.legend(['Share of Firms','Share of Employment'])
            #plt.savefig('pdf_hopehayn.pdf')
            plt.show()
            
            plt.plot(self.prod_grid,self.cdf_stationary)
            plt.plot(self.prod_grid, self.cdf_emp)
            plt.title('Stationary CDF' )
            plt.xlabel('Productivity level')
            plt.ylabel('Cumulative Sum')
            plt.legend(['Share of Firms','Share of Employment'])
            #plt.savefig('cdf_hopehayn.pdf')
            plt.show()       
        
        t1 = time.time()
        print(f'\nTotal Run Time: {t1-t0:.2f} seconds')

# model = Hopenhayn1992(display_plots=False)
# model.solve_model()