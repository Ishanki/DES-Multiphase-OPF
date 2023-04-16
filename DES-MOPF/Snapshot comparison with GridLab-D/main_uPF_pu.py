# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:11:12 2021

@author: Ishanki
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
from pandas import ExcelWriter
from upf_pu import UnbalancedPowerFlow
from time import perf_counter

start = perf_counter()

ST = 1440  # Start Time
FT = 1440   # End Time

# NETWORK_FILE_NAME = 'Network.xlsx'
# df_loadsinfo = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads_info')
# power = df_loadsinfo.set_index(['Bus', 'Time'])['Load'].to_dict()

## Network reduced to 70 buses to enable faster testing 
size = 'largemod'
tfind = 'wtfDY'
NETWORK_FILE_NAME = f'Network_{size}_{tfind}.xlsx'
resultsfile = f'upfslack0_{size}_{tfind}_{FT}.xlsx'
df_loadsinfo = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads_info')
# power = df_loadsinfo.set_index(['Bus', 'Time'])['Load'].to_dict()
# SLACK = 1
# SLACK_ANGLES = [-30, -150, 90]  # In degrees
# PHASE_SHIFT = 0
loads_file_name = 'normal_prof.xlsx'
SLACK = 0
SLACK_ANGLES = [0, -120, 120]  # In degrees
SUBSTATION = 0  #secondary slack where power values are reported
PHASE_SHIFT = -30
PRIMARY_BUSES = [0,999]     #Leave empty if voltages across trafo are the same

df_source = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Source')
df_buses = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Buses')
df_loads = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads')
df_lines = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Lines')
df_linecodes = pd.read_excel(NETWORK_FILE_NAME, sheet_name='LineCodes')
df_transformer = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Transformer')

# house_bus_connections = dict(zip(df_loads.Bus, self.df_loads.phases))
houses = list(df_loads.Bus)
house_num = list(range(1,(len(houses))+1))

load_phase = dict(zip(df_loads.Bus, df_loads.phases))
phases = ['a', 'b', 'c']

m = ConcreteModel()
m.i = Set(initialize=houses)
m.t = RangeSet(ST,FT)
# m.t.pprint()

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
    m.df[n] = m.df[n].iloc[0]
    # print(m.df[n])
    
# Assigning loaded dataframes into dictionaries, now w.r.t house h and time t
# print("data is now loading into loops")
elec_house = {}
for n, h in zip(house_num, houses):
    for t in m.t:
        elec_house[h, t] = round(float(m.df[n][t]),5)         
m.E_load = Param(m.i, m.t, initialize=elec_house)   
# m.E_load = Param(m.i, m.t, initialize=power)   

PF_dict = dict(zip(df_loads.Bus, df_loads.PF))
# print(PF_dict)
# PF = 0.95

Q = {}
for (k1,k2),v in elec_house.items():
    if k2 in m.t:
        Q[k1,k2] = sqrt((v**2)*((1/(PF_dict[k1]**2))-1))
m.Q_load = Param(m.i, m.t, initialize=Q)
# m.Q_load.pprint()     

upf_object = UnbalancedPowerFlow(ST, FT, SLACK, SLACK_ANGLES, PHASE_SHIFT, df_source, df_buses, 
                                  df_loads, df_lines, 
                                  df_linecodes, df_transformer,
                                  primary_buses=PRIMARY_BUSES,
                                  substation=SUBSTATION,
                                  # df_loadsinfo, 
                                  # elec_house,
                                  )
def opf_block(m):
    
    m = upf_object.UPF(power_init=elec_house, load_init=elec_house)
    
    return m

m.OPF = Block(rule=opf_block)

# print(load_phase)
S_BASE = df_transformer.MVA[0]*1000
## Power injections
def P_linking(m,n,p,t):
    for load, phase in load_phase.items():
        if load==n and phase.lower()==p:
            return m.OPF.P[n,p,t] == -m.E_load[load,t]/S_BASE
        elif load==n and phase.lower()!=p:
            return m.OPF.P[n,p,t] == 0
        else:
            continue
    else:
        return Constraint.Skip
m.active_power = Constraint(m.OPF.n, m.OPF.p, m.t, rule=P_linking)
# m.active_power.pprint()
    
# TODO: Power factor and Q linking constraint
def Q_linking(m,n,p,t):
    for load, phase in load_phase.items():
        if load==n and phase.lower()==p:
            # return m.OPF.Q[n,p,t] == 0
            return m.OPF.Q[n,p,t] == -m.Q_load[load,t]/S_BASE
        elif load==n and phase.lower()!=p:
            return m.OPF.Q[n,p,t] == 0
        else:
            continue
    else:
        return Constraint.Skip
m.reactive_power = Constraint(m.OPF.n, m.OPF.p, m.t, rule=Q_linking)

# m.obj = Objective(sense = minimize, expr=1)
m.obj = Objective(sense = minimize, expr=(sum(m.OPF.P[n,p,t] for n in m.OPF.n \
                                              for p in m.OPF.p for t in m.t)))

# solver = SolverFactory('gams')
# results = solver.solve(m, tee=True, solver = 'conopt', 
# # #                         # add_options=['GAMS_MODEL.optfile = 1;',
# # #                         #               '$onecho > conopt.opt', 
# # #                         #               'rtnwma=1.e-6', 
# # #                         #               '$offecho'],
#                         )

solver = SolverFactory('gams')
results = solver.solve(m, tee=True, solver = 'conopt')

# solver = SolverFactory('ipopt')
# options = {}
# options['linear_solver'] = 'ma57'
# results = solver.solve(m, options = options, tee=True,)

stop = perf_counter()
ex_time = stop - start 
print(f"\n**** Time taken to solve model *****: {ex_time}\n")

with open("trblsht.txt", 'w') as f:
    f.write(str(results))
    m.OPF.P.pprint(ostream=f)
    m.OPF.Q.pprint(ostream=f)
    m.OPF.V.pprint(ostream=f)
    m.OPF.theta.pprint(ostream=f)

def VUB_calc(V_dict, theta_dict):
    VUB_dict = {}
    a = -0.5+0.866j
    for n in m.OPF.n:
        for t in range(ST,FT+1):
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

VUB = VUB_calc(m.OPF.V, m.OPF.theta)

resP = pd.DataFrame.from_dict({k: value(v)*S_BASE for k,v in m.OPF.P.items()}, orient="index")
resQ = pd.DataFrame.from_dict({k: value(v)*S_BASE for k,v in m.OPF.Q.items()}, orient="index")
# resV = pd.DataFrame.from_dict({k: value(v) for k,v in m.OPF.V.items()}, orient="index")
# resth = pd.DataFrame.from_dict({k: value(v) for k,v in m.OPF.theta.items()}, orient="index")
resV = {}
resth = {}
for ph in phases:
    resV[ph] = pd.DataFrame.from_dict({(n,p,t): value(v) for (n,p,t),v in m.OPF.V.items()  if p==ph}, orient="index")
    resth[ph] = pd.DataFrame.from_dict({(n,p,t): value(v) for (n,p,t),v in m.OPF.theta.items() if p==ph}, orient="index")
resvub = pd.DataFrame.from_dict({k: value(v) for k,v in VUB.items()}, orient="index")

# resIr = pd.DataFrame.from_dict({k: value(v) for k,v in m.OPF.I_real.items()}, orient="index")
# resIi = pd.DataFrame.from_dict({k: value(v) for k,v in m.OPF.I_imag.items()}, orient="index")

writer = pd.ExcelWriter(resultsfile)
resP.to_excel(writer, sheet_name='P')
resQ.to_excel(writer, sheet_name='Q')
# resV.to_excel(writer, sheet_name='V')
for ph in phases:
    resV[ph].to_excel(writer, sheet_name=f'V_{ph}')
    resth[ph].to_excel(writer, sheet_name=f'Theta_{ph}')
resvub.to_excel(writer, sheet_name='VUB')
# resIr.to_excel(writer, sheet_name='I_real')
# resIi.to_excel(writer, sheet_name='I_imag')
writer.save()
writer.close()
