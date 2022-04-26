# -*- coding: utf-8 -*-
"""
Combining the unbalanced power flow model (NLP) with the DES MILP formulation.

@author: Ishanki
    
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from time import perf_counter
from pandas import ExcelWriter
from MILP_with_complementarity import ResidentialSeasonalDES
from DES_UPF_PU import UnbalancedPowerFlow
from DES_OPF import DES_OPF

# =============================================================================
#                              '''Inputs here'''
# =============================================================================
# house = ['A', 'B', 'C','D','E']
loads_file_name = "profiles.xlsx" 
parameters_file_name = "Parameters.xlsx"  
irrad_file_name = "Parameters.xlsx"
# Battery types given:
battery = ['LI'] #,'SS'] # LI - lithium ion, SS - Sodium-Sulphur
# duration of time interval:
interval = 1  # 0.5 
# State the final time step e.g. 24 or 48:
FT = 24  #48
# Number of seasons to run
SEASONS = 4
# Days in each season, 1-winter, 2-spring, 3-summer, 4-autumn:
d = {1:90, 2:92, 3:92, 4:91} 

# TODO: Activate batteries?
KEEP_BATTERY = 1
KEEP_PV = 1

# Switch OPF functionality off using 0 (when running MILP), else use 1:
# N.B. default solver is CPLEX for MILP, and CONOPT for NLP
KEEP_OPF = 1
SOLVER = "conopt"
KEEP_COMPLEMENTARITY=1

# distribution network
size = 'medmod'
tfind = 'wtfDY'
NETWORK_FILE_NAME = f'Network_{size}_{tfind}.xlsx'
# resultsfile = f'upf_{size}_{tfind}_{FT}.xlsx'

# Network characteristics
SLACK = 0
SLACK_ANGLES = [0, -120, 120]  # in degrees
SUBSTATION = SLACK  # should be slack or can be used as a secondary slack
PHASE_SHIFT = -30   # in degrees
PRIMARY_BUSES = [0,999]  # can leave empty if volts across transformer are same

# Results filenames
if KEEP_OPF == 1:
    if KEEP_COMPLEMENTARITY ==1:
        mstr = "NLP_comp"
    else:
        mstr = "NLPCheck"
else:
    mstr = "MILP"
results_file_name = f"{mstr}_{size}_{tfind}_results" 
results_file_suffix = '.xlsx'

# =============================================================================
#                           '''Data Processing'''
# =============================================================================
df_loadsinfo = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads_info')
df_source = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Source')
df_buses = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Buses')
df_loads = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads')
df_lines = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Lines')
df_linecodes = pd.read_excel(NETWORK_FILE_NAME, sheet_name='LineCodes')
df_transformer = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Transformer')
S_BASE = df_transformer.MVA[0]*1000
# house_bus_connections = dict(zip(df_loads.Bus, self.df_loads.phases))
houses = list(df_loads.Bus)

# this is here to easily loop the data imported from excel
house_num = list(range(1,(len(houses))+1))

start = perf_counter()

# =============================================================================
# '''The residential block'''
# =============================================================================
def MILP(m, binary_cuts=None):
    # Extracting and passing only the data relevant to each season
    def seasonal_data(m,s):
        m.df = {} #represents the dataframe for electricity loads
        m.dfh = {} #represents the dataframe for heating loads
        # Looping through the loads w.r.t each season s and house from excel
        for n, h in zip(house_num, houses):
            ## Elec
            sheet_n1 = (f"Elec_{n}")
            m.df[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n1)
            new_columns = m.df[n].columns.values
            new_columns[0] = 'Season'
            m.df[n].columns = new_columns
            m.df[n].set_index('Season', inplace=True)
            m.df[n] = m.df[n].iloc[s-1]
            ## Heat
            sheet_n2 = (f"Heat_{n}")
            m.dfh[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n2)
            new_columns = m.dfh[n].columns.values
            new_columns[0] = 'Season'
            m.dfh[n].columns = new_columns
            m.dfh[n].set_index('Season', inplace=True)
            m.dfh[n] = m.dfh[n].iloc[s-1]
            string1 = f"this is electricity for season {s} for house {h} "
            string2 = f"this is heat for season {s} for house {h} "
            # print(string1)
            # print(m.df[n])
            #print(b.df[n].get(2))
            
        # Assigning loaded dataframes into dictionaries, now w.r.t house h and time t
        # print("data is now loading into loops")
        m.elec_house = {}
        m.heat_house = {}
        for n, h in zip(house_num, houses):
            for t in range(1,FT+1):
                m.elec_house[h, t] = round(float(m.df[n][t]/interval),5) 
                m.heat_house[h, t] = round(float(m.dfh[n][t]/interval),5)
                    
        #print(m.elec_house['h1', 34])
        #print(m.heat_house['h2',34]) 
        
        # Loading other time-dependent parameters
        m.dfi = pd.read_excel(irrad_file_name, sheet_name = "Irradiance")
        m.dfi.set_index("Irrad", inplace = True)
        m.dfi = m.dfi.iloc[s-1]
        string3 = "this is irradiance data for season "
        #print(string3 + str(s))
        # print(m.dfi)
        
        m.irrad = {}
        for t in range(1,FT+1):
            m.irrad[t] = float(m.dfi[t])
        # print(m.irrad)
    
        m.days = d[s]
        
        # These dataframes are fed directly into the ResidentialSeasonalDES class
        df_scalars = pd.read_excel(parameters_file_name, sheet_name = "Res_Scalars")
        #print(df_scalars)
        df_roof = pd.read_excel(parameters_file_name, sheet_name = "Roof_areas_res")
        df_batteries = pd.read_excel(parameters_file_name, sheet_name = "batteries")
        df_volume = pd.read_excel(parameters_file_name, sheet_name = "Stor_vol_res")
        
        # The object m.full_model is created from the ResidentialSeasonalDES class
        m.full_model = ResidentialSeasonalDES(house=houses, 
                                          df=m.df, 
                                          days=m.days,
                                          interval=interval, 
                                          ft=FT, 
                                          irrad=m.irrad,
                                          df_scalars=df_scalars,
                                          df_roof=df_roof,
                                          elec_house=m.elec_house,
                                          heat_house=m.heat_house,
                                          df_batteries=df_batteries,
                                          df_volume=df_volume, 
                                          battery=battery,
                                          df_loads=df_loads,
                                          SEASONS=SEASONS,
                                          KEEP_BATTERY=KEEP_BATTERY,
                                          KEEP_PV=KEEP_PV,
                                          )
        
        # Assigning the DES_MILP method in the full_model object to the Pyomo model m
        m = m.full_model.DES_MILP()
        
        # Deactivating the individual objectives in each block
        m.objective.deactivate()  
        
        # This is the free variable for total cost which the objective minimises
        m.cost = Var(bounds = (None, None))
        # m.cost = Var(bounds = (None, 7000)) #Octeract bound changes when this is included
        
        #This is the objective function rule that combines the costs for that particular season
        def rule_objective(m):
            expr = 0
            expr += (m.annual_cost_grid + m.carbon_cost + m.annual_inv_PV + \
                     m.annual_inv_B + m.annual_inv_S + \
                         m.annual_oc_PV + m.annual_oc_b + m.annual_oc_S \
                             - m.export_income - m.gen_income)
            #expr += m.annual_cost_grid
            #expr += (sum(m.E_discharge[i,t,c] for i in m.i for t in m.t for c in m.c))
            return m.cost == expr
        m.obj_constr = Constraint(rule = rule_objective)
    
        
        # This function returns the model m to be used later within the code
        return m
    
    def DCOPF_block(m,s):
        # DC OPF for MILP
        m = opf_model.DCOPF()
        return m
    
    # Initialising the OPF class for DC approximation
    opf_model = DES_OPF(st=1, ft=FT,
                        I_MAX=None,
                        PHASE_SHIFT=PHASE_SHIFT,
                        primary_buses=PRIMARY_BUSES,
                      df_source=df_source,
                        slack=SLACK,
                        substation=SUBSTATION,
                        df_transformer=df_transformer,
                        df_lines=df_lines,
                        df_loads=df_loads,
                        df_buses=df_buses,
                        df_linecodes=df_linecodes,
                        )
    
    # Assigning the function to a Block so that it loops through all the seasons
    m.DES_res = Block(m.S, rule=seasonal_data)
    m.DCOPF = Block(m.S, rule=DCOPF_block)
    
    count = 0
    m.map_season_to_count = dict()
    m.first_season = None
    for i in m.S:
        m.map_season_to_count[count] = i
        if count == 0:
            m.first_season = i
        count += 1
    
    # Linking the capacities of all the DES technologies used within the model
    def linking_PV_panels_residential_rule(m,season,house):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].panels_PV[house] == m.DES_res[previous_season].panels_PV[house]
      
    m.PV_linking_res = Constraint(m.S, houses, 
                                  rule = linking_PV_panels_residential_rule)
    
    def boiler_capacities_residential_rule(m,season,house):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].max_H_b[house] == m.DES_res[previous_season].max_H_b[house]
    
    m.boiler_linking_res = Constraint(m.S, houses, 
                                      rule = boiler_capacities_residential_rule)
    
    def battery_capacities_residential_rule(m,season,house,battery):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].storage_cap[house,battery] == m.DES_res[previous_season].storage_cap[house,battery]
    
    m.battery_linking_res = Constraint(m.S, houses, battery, 
                                       rule = battery_capacities_residential_rule)
    
    # Ensure that powers here are given in p.u. by dividing with S_BASE
    def DC_P_linking(m,s,n,t):
        for load in houses:
            if load==n:
                return ((m.DES_res[s].E_PV_sold[n,t] - m.DES_res[s].E_grid[n,t] \
                    - sum(m.DES_res[s].E_grid_charge[n,t,b] for b in battery)))/S_BASE == m.DCOPF[s].P[n,t]
            else:
                continue
        else:
            return Constraint.Skip
    m.DC_active_power = Constraint(m.S, m.DCOPF[1].n, m.t, rule=DC_P_linking)
    
    ## Deactivating constraints related to complementarity
    for s in m.S:
        m.DES_res[s].SC9c.deactivate()
        m.DES_res[s].CP0a.deactivate()
        m.DES_res[s].CP0b.deactivate()
        
    #This is the objective function combining residential costs for all seasons
    m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:]))
    
    return m


def NLP(m):
    # Deactivating DC constraints before adding AC constraints
    for s in m.S:
        m.DCOPF[s].DC1.deactivate()
        m.DCOPF[s].DC2.deactivate()
        m.DCOPF[s].DC3.deactivate()
        m.DCOPF[s].DC4.deactivate()
        m.DCOPF[s].theta.unfix()
    m.DC_active_power.deactivate()
    
    # Fixing binary variables for NLP (and capacities for NLPChecks if needed)
    for s in m.S:
        for h in m.DES_res[s].i:
            # m.DES_res[s].panels_PV[h].fix()
            # m.DES_res[s].max_H_b[h].fix()
            for t in m.DES_res[s].t:
                m.DES_res[s].X[h,t].fix()
                # Operation
                # m.DES_res[s].E_PV_sold[h,t].fix()
                for c in m.DES_res[s].c:
                    # m.DES_res[s].storage_cap[h,c].fix()
                    m.DES_res[s].Q[h,t,c].fix()
                    m.DES_res[s].W[h,c].fix()
                    # Operation
                    # m.DES_res[s].E_PV_charge[h,t,c].fix()
    
    ## Active power (kW) initialisation values for each season, conversion to p.u. happens in class
    power_init = {}
    for s in m.S:
        values = {}
        for n in houses:
            for t in m.t:
                values[n,t] = (m.DES_res[s].E_PV_sold[n,t].value \
                               - m.DES_res[s].E_grid[n,t].value \
                         - sum(m.DES_res[s].E_grid_charge[n,t,b].value \
                               for b in battery))
        power_init[s] = values
    # print(power_init)
    # print(" ")
    
    ## Loads (kW) in each season to initialise reactive power, conversion to p.u. happens in class
    load_init = {}
    for s in m.S:
        load_init[s] = m.DES_res[s].elec_house
        
    ## Creating the opf_model object from the DES_OPF class
    upf_object = UnbalancedPowerFlow(FT, SLACK, SLACK_ANGLES, PHASE_SHIFT, 
                                     df_source, df_buses, 
                                     df_loads, df_lines, 
                                      df_linecodes, 
                                     df_transformer,
                                     primary_buses=PRIMARY_BUSES,
                                     substation=SUBSTATION,
                                     # df_loadsinfo,
                                     )
           
    def OPF_block(m,s): 
        
        ## Assigning the OPF method in opf_model to the Pyomo model m
        m = upf_object.UPF(power_init=power_init[s], 
                           load_init=load_init[s])
        
        # This function also returns the Pyomo model 
        return m
    
    m.OPF_res = Block(m.S, rule=OPF_block)
    
    # Reactivating the nonlinear equations
    # for s in m.S:
    #     m.DES_res[s].INV6.activate()
        # deactivating the current constraint if required
        # m.OPF_res[s].C12.deactivate()
    
    load_phase = dict(zip(df_loads.Bus, df_loads.phases))
    
    ## Ensure that powers here are given in p.u. by dividing with S_BASE
    def P_linking(m,s,n,p,t):
        for load, phase in load_phase.items():
            if load==n and phase.lower()==p:
                return ((m.DES_res[s].E_PV_sold[n,t] - m.DES_res[s].E_grid[n,t] \
                    - sum(m.DES_res[s].E_grid_charge[n,t,b] for b in battery)))/S_BASE == m.OPF_res[s].P[n,p,t]
            elif load==n and phase.lower()!=p:
                return m.OPF_res[s].P[n,p,t] == 0
            else:
                continue
        else:
            return Constraint.Skip
    m.active_power = Constraint(m.S, m.OPF_res[1].n, m.OPF_res[1].p, m.t, rule=P_linking)
    # m.active_power.pprint()
        
    # TODO: Power factor and Q linking constraint
    def Q_linking(m,s,n,p,t):
        for load, phase in load_phase.items():
            if load==n and phase.lower()==p:
                # return m.OPF_res[s].Q[n,p,t] == 0
                return m.OPF_res[s].Q[n,p,t] == -m.DES_res[s].Q_load[load,t]/S_BASE
            elif load==n and phase.lower()!=p:
                return m.OPF_res[s].Q[n,p,t] == 0
            else:
                continue
        else:
            return Constraint.Skip
    m.reactive_power = Constraint(m.S, m.OPF_res[1].n, m.OPF_res[1].p, m.t, rule=Q_linking)
    
    return m

def VUB_calc(V_dict, theta_dict):
    VUB_dict = {}
    a = -0.5+0.866j
    for n in m.OPF_res[1].n:
        for t in range(1,FT+1):
            Va = V_dict[n,'a',t].value*(np.cos(theta_dict[n,'a',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'a',t].value)))
            Vb = V_dict[n,'b',t].value*(np.cos(theta_dict[n,'b',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'b',t].value)))
            Vc = V_dict[n,'c',t].value*(np.cos(theta_dict[n,'c',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'c',t].value)))
            V1 = (Va+a*Vb+a**2*Vc)/3
            # print(V1)
            V2 = (Va+a**2*Vb+a*Vc)/3
            # print(V2)
            VUB_cmplx = V2/V1
            VUB = np.sqrt(np.real(VUB_cmplx)**2+np.imag(VUB_cmplx)**2)
            VUB_dict[n,t] = VUB*100
    return VUB_dict

# run_alg = True
start = perf_counter()

# import sys 
# stdoutOrigin=sys.stdout 
# sys.stdout = open("log.txt", "w")


# Creating new Concrete Model per run
m = ConcreteModel()
# State the total amount of timestamps here:
m.t = RangeSet(FT) 
# Number of seasons, adjust the number accordingly:
m.S = RangeSet(SEASONS) 

# Solve MILP
m = MILP(m)
solver = SolverFactory('gams')
results = solver.solve(m, tee=True, 
                solver = 'cplex', add_options=['option optcr=0.001;'])
with open(f'resultsCPLEX_{size}_{tfind}.txt',"w") as f:
    f.write(str(results))


# Solve NLP
EPSILON = 0.1
counter = 1
if KEEP_OPF == 1:
    m = NLP(m)
    ## Solving with fixed binary topology first
    if SOLVER.lower() == 'conopt':
        solver = SolverFactory('gams')
        results = solver.solve(m, tee=True, solver = 'conopt',
                    add_options=['option reslim=100000;'])
    elif SOLVER.lower() == 'ipopt':
        solver = SolverFactory('ipopt')
        options = {}
        options['linear_solver'] = 'ma57'
        results = solver.solve(m, options = options, tee=True,)
        results = solver.solve(m, tee=True)
    else:
        print("this solver is unavailable")
        
    if KEEP_COMPLEMENTARITY == 1:
    ## Deactivating and activating constraints related to complementarity
        for s in m.S:
            # Q constraints
            m.DES_res[s].SC9a.deactivate()
            m.DES_res[s].SC9b.deactivate()
            m.DES_res[s].SC9c.activate()
            # X constraints
            m.DES_res[s].bfg_constraint.deactivate()
            m.DES_res[s].stg_constraint.deactivate()
            m.DES_res[s].CP0a.activate()
            m.DES_res[s].CP0b.activate()
        
        while EPSILON >= 1e-7:
            print(f"\n------Iteration {counter}------\n")
            for s in m.S:
                m.DES_res[s].EPSILON = EPSILON
    
            if SOLVER.lower() == 'conopt':
                solver = SolverFactory('gams')
                results = solver.solve(m, tee=True, solver = 'conopt',
                            add_options=['option reslim=100000;'])
            elif SOLVER.lower() == 'ipopt':
                solver = SolverFactory('ipopt')
                options = {}
                options['linear_solver'] = 'ma57'
                results = solver.solve(m, options = options, tee=True,)
                results = solver.solve(m, tee=True)
            else:
                print("this solver is unavailable")
                
            if results.solver.termination_condition == TerminationCondition.locallyOptimal:
                EPSILON = EPSILON/10
            else:
                EPSILON = EPSILON*15
            # EPSILON = EPSILON/10
            counter +=1
    
    with open(f'results_{mstr}_{size}_{tfind}.txt',"w") as f:
        f.write(str(results))

    VUB = {}
    for s in m.S:
        VUB[s] = VUB_calc(m.OPF_res[s].V, m.OPF_res[s].theta)
    
    
# Some prints to sanity check the costs
print(f"\n")
annual_grid = sum(m.annual_cost_grid.value for m in m.DES_res[:])
print("annual_grid_cost = " + str(annual_grid))
carb_grid = sum(m.carbon_cost.value for m in m.DES_res[:])
print("grid_carbon_cost = " + str(carb_grid))
annual_PV_inv_cost = sum(m.annual_inv_PV.value for m in m.DES_res[:])
print("annual_PV_inv_cost = " + str(annual_PV_inv_cost))
annual_PV_op_cost = sum(m.annual_oc_PV.value for m in m.DES_res[:])
print("annual_PV_op_cost = " + str(annual_PV_op_cost))
annual_boiler_inv_cost = sum(m.annual_inv_B.value for m in m.DES_res[:])
print("annual_boiler_inv_cost = " + str(annual_boiler_inv_cost))
annual_boiler_op_cost = sum(m.annual_oc_b.value for m in m.DES_res[:])
print("annual_boiler_op_cost = " + str(annual_boiler_op_cost))
annual_batt_inv_cost = sum(m.annual_inv_S.value for m in m.DES_res[:])
print("annual_battery_inv_cost = " + str(annual_batt_inv_cost))
annual_batt_op_cost = sum(m.annual_oc_S.value for m in m.DES_res[:])
print("annual_battery_op_cost = " + str(annual_batt_op_cost))
annual_inc = sum(m.export_income.value for m in m.DES_res[:])
print("export_income = " + str(annual_inc))
annual_FIT = sum(m.gen_income.value for m in m.DES_res[:])
print("annual_FIT = " + str(annual_FIT))

stop = perf_counter()
ex_time = stop - start 

print(f"\n****Total time*****: {ex_time}\n")

cost_dict = {'Objective':m.obj.expr(),
             "annual_grid":annual_grid,
             "carb_grid":carb_grid,
             "annual_PV_inv_cost":annual_PV_inv_cost,
             "annual_PV_op_cost":annual_PV_op_cost,
             "annual_boiler_inv_cost":annual_boiler_inv_cost,
             "annual_boiler_op_cost":annual_boiler_op_cost,
             "annual_batt_inv_cost":annual_batt_inv_cost,
             "annual_batt_op_cost":annual_batt_op_cost,
             "annual_export_inc":annual_inc,
             "annual_FIT":annual_FIT,
             "time taken":ex_time,
             'solver termination': results.solver.termination_condition,
             }

# sys.stdout.close()
# sys.stdout=stdoutOrigin

for s in m.S:
    rdf_cost = pd.DataFrame.from_dict(cost_dict, orient="index", columns=["value"])
    #for i,t in zip(house,m.t):
    '''residential results'''
    E_grid_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_grid.items()}
    rdf_result1 = pd.DataFrame.from_dict(E_grid_res_data, orient="index", columns=["variable value"])
    
    panels_PV_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].panels_PV.items()}
    rdf_result2 = pd.DataFrame.from_dict(panels_PV_res_data, orient="index", columns=["variable value"])
    
    E_PV_sold_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_sold.items()}
    rdf_result3 = pd.DataFrame.from_dict(E_PV_sold_res_data, orient="index", columns=["variable value"])
    
    E_PV_used_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_used.items()}
    rdf_result4 = pd.DataFrame.from_dict(E_PV_used_res_data, orient="index", columns=["variable value"])
    
    Max_H_b_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].max_H_b.items()}
    rdf_result5 = pd.DataFrame.from_dict(Max_H_b_res_data, orient="index", columns=["variable value"])
    
    Storage_cap_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].storage_cap.items()}
    rdf_result8 = pd.DataFrame.from_dict(Storage_cap_res_data, orient="index", columns=["variable value"])
    
    Q_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q.items()}
    rdf_result9 = pd.DataFrame.from_dict(Q_res_data, orient="index", columns=["variable value"])
    
    Storage_volume_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].volume.items()}
    rdf_result14 = pd.DataFrame.from_dict(Storage_volume_res, orient="index", columns=["variable value"])
    
    X_res_data = {(i, t, v.name): value(v) for (i,t), v in m.DES_res[s].X.items()}
    rdf_result16 = pd.DataFrame.from_dict(X_res_data, orient="index", columns=["variable value"])
    
    type_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].W.items()}
    rdf_result17 = pd.DataFrame.from_dict(type_res_data, orient="index", columns=["variable value"])
    
    # New additions and OPF related exports
    # Inv_cap_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].inv_cap.items()}
    # rdf_result18 = pd.DataFrame.from_dict(Inv_cap_res, orient="index", columns=["variable value"])
    
    # PInv_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].P_inv.items()}
    # rdf_result19 = pd.DataFrame.from_dict(PInv_res_data, orient="index", columns=["variable value"])
    
    
    if KEEP_OPF == 1:
        Q_gen_res = {(i,t): v for (i,t), v in m.DES_res[s].Q_load.items()}
        rdf_result21 = pd.DataFrame.from_dict(Q_gen_res, orient="index", columns=["variable value"])
        
        V_res = {(n,p,t, v.name): value(v) for (n,p,t), v in m.OPF_res[s].V.items()}
        rdf_result22 = pd.DataFrame.from_dict(V_res, orient="index", columns=["variable value"])
        
        Theta_res = {(n,p,t, v.name): value(v) for (n,p,t), v in m.OPF_res[s].theta.items()}
        rdf_result23 = pd.DataFrame.from_dict(Theta_res, orient="index", columns=["variable value"])
        
        Q_res = {(n,p,t, v.name): value(v)*S_BASE for (n,p,t), v in m.OPF_res[s].Q.items()}
        rdf_result24 = pd.DataFrame.from_dict(Q_res, orient="index", columns=["variable value"])
        
        P_res = {(n,p,t, v.name): value(v)*S_BASE for (n,p,t), v in m.OPF_res[s].P.items()}
        rdf_result25 = pd.DataFrame.from_dict(P_res, orient="index", columns=["variable value"])
        
        # I_res = {(n,m,t, v.name): value(v) for (n,m,t),v in m.OPF_res[s].current_sqr.items()}
        # rdf_result26 = pd.DataFrame.from_dict(I_res, orient="index", columns=["variable value"])
        
        VUB_res = {(n,t): v for (n,t), v in VUB[s].items()}
        rdf_result27 = pd.DataFrame.from_dict(VUB_res, orient="index", columns=["variable value"])
        
    
    E_PVch_res = {}
    rdf_EPVch = {}
    E_stored_res = {}
    rdf_stored = {}
    E_gch_res = {}
    rdf_gridcharge = {}
    E_chg_res = {}
    rdf_chg = {}
    E_dsch_res = {}
    rdf_dsch = {}
    
    for bat_num in range(len(battery)):
        E_PVch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_PV_charge.items() if c == battery[bat_num]} 
        rdf_EPVch[bat_num] = pd.DataFrame.from_dict(E_PVch_res[bat_num], orient="index", columns=["variable value"])
        E_stored_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items() if c == battery[bat_num]}
        rdf_stored[bat_num] = pd.DataFrame.from_dict(E_stored_res[bat_num], orient="index", columns=["variable value"])
        E_gch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_grid_charge.items() if c == battery[bat_num]}
        rdf_gridcharge[bat_num] = pd.DataFrame.from_dict(E_gch_res[bat_num], orient="index", columns=["variable value"])
        E_chg_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_charge.items() if c == battery[bat_num]}
        rdf_chg[bat_num] = pd.DataFrame.from_dict(E_chg_res[bat_num], orient="index", columns=["variable value"])
        E_dsch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_discharge.items() if c == battery[bat_num]}
        rdf_dsch[bat_num] = pd.DataFrame.from_dict(E_dsch_res[bat_num], orient="index", columns=["variable value"])
    
        
    Results_file_name1 = results_file_name+str(s)+results_file_suffix
    writer = ExcelWriter(Results_file_name1)
    rdf_cost.to_excel(writer, "Costs")
    rdf_result8.to_excel(writer, 'Res_storage_cap')
    rdf_result14.to_excel(writer, 'Res_stor_vol')
    rdf_result2.to_excel(writer, 'Res_Panels_PV')
    rdf_result5.to_excel(writer, 'Res_max_H_b')
    rdf_result1.to_excel(writer,'Res_E_grid')
    rdf_result3.to_excel(writer, 'Res_E_PV_sold')
    rdf_result4.to_excel(writer, 'Res_E_PV_used')
    
    for bat_num in range(len(battery)):
        for k, v in rdf_EPVch[bat_num].items():
            if sum(v) != 0:
                rdf_EPVch[bat_num].to_excel(writer, f"Res_E_PV_ch_{battery[bat_num]}")
        
        for k, v in rdf_stored[bat_num].items():
            if sum(v) != 0:
                rdf_stored[bat_num].to_excel(writer, f"Res_E_stored_{battery[bat_num]}")
        
        for k, v in rdf_gridcharge[bat_num].items():
            if sum(v) != 0:
                rdf_gridcharge[bat_num].to_excel(writer, f"Res_E_grd_ch_{battery[bat_num]}")
        
        for k, v in rdf_chg[bat_num].items():
            if sum(v) != 0:
                rdf_chg[bat_num].to_excel(writer, f"Res_E_charge_{battery[bat_num]}")
        
        for k, v in rdf_dsch[bat_num].items():
            if sum(v) != 0:
                rdf_dsch[bat_num].to_excel(writer, f"Res_E_disch_{battery[bat_num]}")
    
    
    if KEEP_OPF == 1:
        # rdf_result18.to_excel(writer, 'Inv_Cap')
        # rdf_result19.to_excel(writer, 'P_inv')
        rdf_result21.to_excel(writer, 'Q_gen')
        rdf_result22.to_excel(writer, 'V_OPF')
        rdf_result23.to_excel(writer, 'Angle_OPF')
        rdf_result25.to_excel(writer, 'P_node')
        rdf_result24.to_excel(writer, 'Q_node')
        # rdf_result26.to_excel(writer, 'I_sqr_node')
        rdf_result27.to_excel(writer, 'VUB')
        
    rdf_result9.to_excel(writer, 'Res_Q')
    rdf_result16.to_excel(writer, 'Res_X')

    writer.save()
    writer.close()
    

