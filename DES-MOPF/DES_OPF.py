# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:30:56 2020

@author: ishan
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from pandas import ExcelWriter

class DES_OPF(object):
    def __init__(self, st, ft, \
                 I_MAX,
                 PHASE_SHIFT,
                 df_source, slack,
                 substation,
                 df_transformer,
                 df_lines,
                 df_loads,
                 df_buses,
                 df_linecodes,
                 primary_buses,
                 ):
        self.st = st #house names
        self.ft = ft
        self.PHASE_SHIFT = PHASE_SHIFT
        self.df_source = df_source
        self.slack = slack
        self.secondary_slack = substation
        self.df_transformer = df_transformer
        self.df_lines = df_lines
        self.df_loads = df_loads
        self.df_buses = df_buses
        self.df_linecodes = df_linecodes
        self.primary_buses = primary_buses
        
        self.nodes = [i for i in self.df_buses.Busname]
        self.lines = list(zip(self.df_lines.Bus1, self.df_lines.Bus2))
        self.gen = self.df_loads.Bus.to_list()
        self.V_BASE_LL = self.df_transformer.kV_pri[0]  # Source line voltage base in V
        self.PU_SOURCE = self.df_source.iat[0,1]
        self.S_BASE = self.df_transformer.MVA[0]*1000
        # V_SLACK = self.V_SRC_BASE_LL*self.PU_BASE
        ## Load side has a wye configuration
        # V_BASE_LOAD_LL = self.df_transformer.kV_sec[0]*1e3   # Line base voltage in V 
        self.V_LOAD_LL = self.df_transformer.kV_sec[0] # Phase base voltage in V
        self.V_UB = self.df_source.iat[5,1]
        self.V_LB = self.df_source.iat[6,1]
        # print(self.V_BASE_LL, self.PU_SOURCE, self.S_BASE, self.V_LOAD_LL)
        print(f'\n')
        # print(self.V_UB, self.V_LB)
        
        self.V1_BASE = self.V_BASE_LL/sqrt(3)
        self.I1_BASE = self.S_BASE/(sqrt(3)*self.V_BASE_LL)
        self.Z1_BASE = self.V_BASE_LL**2*1000/self.S_BASE
        print(self.V1_BASE, self.I1_BASE, self.Z1_BASE)
        
        self.TRF_RATIO = self.V_BASE_LL/self.V_LOAD_LL
        print(self.TRF_RATIO)
        
        self.V2_BASE = self.V1_BASE/self.TRF_RATIO
        self.I2_BASE = self.TRF_RATIO*self.I1_BASE
        self.Z2_BASE = self.Z1_BASE/(self.TRF_RATIO**2)
        print(self.V2_BASE, self.I2_BASE, self.Z2_BASE)
        
        self.I_MAX = I_MAX
        
        self.admittance_calcs()
        print("Starting OPF...")
        print(f'\n')
        
    def admittance_calcs(self):
        R1 = {}
        X1 = {}
        trf_keys = []
        self.a = {(n,m):1 for n in self.nodes for m in self.nodes}
        for i in range(len(self.lines)):
            for line in self.df_linecodes.Name:
                if self.df_lines.LineCode[i] == line:
                    val = self.df_linecodes.index[self.df_linecodes.Name==line].values.astype(int)[0]
                    # Converting from Ohms/km to Ohms by multiplying R or X with length
                    if self.lines[i][0] in self.primary_buses and self.lines[i][1] in self.primary_buses:
                        R1[self.lines[i]] = self.df_linecodes.R1[val]*self.df_lines.Length[i]/(1000*self.Z1_BASE)
                        X1[self.lines[i]] = self.df_linecodes.X1[val]*self.df_lines.Length[i]/(1000*self.Z1_BASE)
                    else:
                        R1[self.lines[i]] = self.df_linecodes.R1[val]*self.df_lines.Length[i]/(1000*self.Z2_BASE)
                        X1[self.lines[i]] = self.df_linecodes.X1[val]*self.df_lines.Length[i]/(1000*self.Z2_BASE)
        
        ## This is commented out for the small case study as we only evaluate the secondary side
        for trf in range(len(self.df_transformer)):
            ## Transformer impedance in p.u.
            Ztseries = complex(self.df_transformer['% resistance'][trf]/1e2,
                                self.df_transformer['%XHL'][trf]/1e2)
            P_bus = self.df_transformer.iloc[trf]['bus1']    # Primary bus
            S_bus = self.df_transformer.iloc[trf]['bus2']    # Secondary bus
            P_conn = self.df_transformer.iloc[trf]['Conn_pri'].lower()  # Primary connection
            S_conn = self.df_transformer.iloc[trf]['Conn_sec'].lower()  # Secondary connection
            # print(P_bus, S_bus)
            R1[P_bus, S_bus] = self.df_transformer['% resistance'][trf]/1e2
            X1[P_bus, S_bus] = self.df_transformer['%XHL'][trf]/1e2
            trf_keys.append((P_bus, S_bus))
            y_series = 1/Ztseries
            # print(y_series)
            self.a[P_bus,S_bus] = self.TRF_RATIO*np.exp(1j*np.radians(self.PHASE_SHIFT))
        
        self.R1=R1
        self.X1=X1
        self.trf_keys=trf_keys
        
    def OPF(self,power_init=None,load_init=None):
        
        model = ConcreteModel()
        
        model.n = Set(initialize = self.nodes)
        model.m = Set(initialize = model.n)
        model.t = RangeSet(self.st, self.ft, doc= 'periods/timesteps')

        model.line_R = Param(model.n, model.m, initialize = self.R1, default=0)
        model.line_X = Param(model.n, model.m, initialize = self.X1, default=0)
        
        '''Constructing the admittance matrix'''
        def branch_series_admittance(model,n,m):
            if (model.line_R[n,m]**2+model.line_X[n,m]**2) !=0:
                return complex((model.line_R[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)),\
                           -(model.line_X[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)))
            else:
                return 0
        model.y_series = Param(model.n,model.m, initialize = branch_series_admittance)
        
        y_series_mags_sqr = {}
        for k, v in model.y_series.items():
            if v != 0:
                y_series_mags_sqr[k] = v.real**2 + v.imag**2
            else:
                y_series_mags_sqr[k] = 0
        model.y_branch_magsqr = Param(model.n, model.m, initialize = y_series_mags_sqr)
        # model.y_branch_magsqr.pprint()
        
        Y = {}
        for k,v in model.y_series.items():
            Y[k] = v
        #print(Y)
        
        diags = {n:(n,n) for n in self.nodes}
        non_diags = {(n,k):0 for n in self.nodes for k in self.nodes if n!=k}
        # print(non_diags)
        
        ndiags = []
        for i in self.trf_keys:
            ndiags.append(i)
            ndiags.append((i[1], i[0]))
        ndiags.extend(self.lines)
        reverse_lines = [(sub[1], sub[0]) for sub in self.lines]
        ndiags.extend(reverse_lines)
        # print(ndiags)
        
        y_diags = {}
        for k, (v1,v2) in diags.items():
            # print(k)
            y_diags[k,k] = sum(v*1/(self.a[k1,k2]*np.conj(self.a[k1,k2])) for (k1,k2),v in Y.items() if k==k1 or k==k2)
        #print("DIAGONALS")
        # print(y_diags)
        
        
        y_nd = non_diags
        for key in ndiags:
        # for key,v in non_diags.items():
            y_nd[key[0],key[1]] = - sum(v*1/np.conj(self.a[k1,k2]) for (k1,k2),v in Y.items() if key[0]==k1 and key[1]==k2) - \
                sum(v*1/self.a[k1,k2] for (k1,k2),v in Y.items() if key[0]==k2 and key[1]==k1)
        #print("NON-DIAGONALS")
        # print(y_nd)
            
        Admittance = {**y_diags,**y_nd}

        Conductance = {}
        Susceptance = {}
        for k, v in Admittance.items():
            Conductance[k]= v.real
            Susceptance[k]= v.imag
        
        G = Conductance
        B = Susceptance
        
        # =============================================================================
        # Variable inits
        # =============================================================================
        P_init = {(n,t):0 for n in self.nodes for t in model.t}
        Q_init = {(n,t):0 for n in self.nodes for t in model.t}
        PF_dict = dict(zip(self.df_loads.Bus, self.df_loads.PF))
        
        if power_init!=None:
            for load in self.gen:
                for (bus,time), val in power_init.items():
                    if load == bus:
                        P_init[bus,time] = val/self.S_BASE
        
        if load_init!=None:
            for load in self.gen:
                for (bus,time), v in load_init.items():
                    if load == bus:
                        if v!=0:
                            Q_init[bus,time] = \
                                -np.sqrt((v**2)*((1/(PF_dict[load]**2))-1))\
                                    /self.S_BASE
        
        for n in self.nodes:
            for t in model.t:
                if n==self.secondary_slack:
                    P_init[n,t] = sum(P_init[n,t] for n in self.nodes \
                                        if n!=self.slack)*-1
                    Q_init[n,t] = sum(Q_init[n,t] for n in self.nodes \
                                        if n!=self.slack)*-1
        
        theta_init = {}
        for n in self.nodes:
            for t in model.t:
                if n in self.primary_buses:
                    theta_init[n,t] = np.radians(0)
                    theta_init[n,t] = np.radians(0)
                    theta_init[n,t] = np.radians(0)
                else:
                    theta_init[n,t] = np.radians(self.PHASE_SHIFT)
                    theta_init[n,t] = np.radians(self.PHASE_SHIFT)
                    theta_init[n,t] = np.radians(self.PHASE_SHIFT)
        
        with open('init_check.txt','w') as f:
            for k, v in P_init.items():
                f.write(f'{k}: {v*self.S_BASE}\n')
            f.write('\n\n')
            for k, v in Q_init.items():
                f.write(f'{k}: {v*self.S_BASE}\n')
            f.write('\n\n')
        
        # =============================================================================
        # Variables and Constraints
        # =============================================================================
        
        model.V = Var(model.n, model.t, bounds = (None,None), doc = 'node voltage', initialize = self.PU_SOURCE)
        model.P = Var(model.n, model.t,  bounds = (None,1e5), doc = 'node active power', initialize = P_init)
        model.Q = Var(model.n, model.t,  bounds = (None,1e5), doc = 'node reactive power', initialize = Q_init)
        model.theta = Var(model.n, model.t, bounds = (None,None), doc = 'voltage angle', initialize = theta_init)
        model.current_sqr = Var(model.n, model.m, model.t, bounds = (None,None), initialize=0)
        model.dummy = Var(model.n, model.t, bounds = (None,None), 
                          doc = 'dummy var to give DoF to IPOPT', initialize = 0)
        
        ## This is a dummy constraint to give an extra DoF to allow IPOPT to solve
        def dummyc(model, t):
            return sum(model.dummy[n,t] for n in model.n) >= 1
        model.DC1 = Constraint(model.t, rule=dummyc)
        
        def P_balance(model,n,t):
            return model.P[n,t] == model.V[n,t]*sum(model.V[m,t]*((G[n,m]*cos(model.theta[n,t]-model.theta[m,t])) \
                                         + (B[n,m]*sin(model.theta[n,t]-model.theta[m,t]))) for m in model.m)
        model.C1 = Constraint(model.n,model.t, rule = P_balance)
        
        def Q_balance(model,n,t):
            return model.Q[n,t] ==  model.V[n,t]*sum(model.V[m,t]*((G[n,m]*sin(model.theta[n,t]-model.theta[m,t])) \
                                         - (B[n,m]*cos(model.theta[n,t]-model.theta[m,t]))) for m in model.m)
        model.C2 = Constraint(model.n,model.t, rule = Q_balance)
        
        for t in model.t:
            model.V[self.slack,t].fix(self.PU_SOURCE)
        
        def V_bus_upper(model,n,t):
            if n != self.slack:
                # return Constraint.Skip
                return model.V[n,t] <= self.V_UB/self.V2_BASE
            else:
                return model.V[n,t] == self.PU_SOURCE
        model.C3 = Constraint(model.n,model.t, rule = V_bus_upper)
        
        def V_bus_lower(model,n,t):
            if n != self.slack:
                # return Constraint.Skip
                return model.V[n,t] >= self.V_LB/self.V2_BASE
            else:
                return model.V[n,t] == self.PU_SOURCE
        model.C4 = Constraint(model.n,model.t, rule = V_bus_lower)
        
        for t in model.t:
            model.theta[self.slack,t].fix(np.radians(0))
        
        def theta_upper(model,n,t):
            if n != self.slack:
                return model.theta[n,t] <= np.pi
            else:
                return model.theta[n,t] == 0
        model.C5 = Constraint(model.n,model.t, rule = theta_upper)
        
        def theta_lower(model,n,t):
            if n != self.slack:
                return model.theta[n,t] >= (-np.pi)
            else:
                return model.theta[n,t] == 0
        model.C6 = Constraint(model.n,model.t, rule = theta_lower)
        
        def Non_gen_P(model,n,t):
            if n !=self.secondary_slack and n not in self.gen:
                return model.P[n,t] == 0
            else:
                return Constraint.Skip
        model.C7 = Constraint(model.n,model.t, rule = Non_gen_P)
        
        def Non_gen_Q(model,n,t):
            if n != self.secondary_slack and n not in self.gen:
                return model.Q[n,t] == 0
            else:
                return Constraint.Skip
        model.C8 = Constraint(model.n,model.t, rule = Non_gen_Q)
        
        # New current constraint using series branch admittance
        def current_equality(model,n,m,t):
            if value(model.y_branch_magsqr[n,m]) != 0:
                return model.current_sqr[n,m,t] == (((model.V[n,t]*cos(model.theta[n,t])) - (model.V[m,t]*cos(model.theta[m,t])))**2 \
                    + ((model.V[n,t]*sin(model.theta[n,t])) - (model.V[m,t]*sin(model.theta[m,t])))**2) \
                    * model.y_branch_magsqr[n,m]
            else:
                return Constraint.Skip
        model.C11 = Constraint(model.n, model.m, model.t, rule=current_equality)
        
        if self.I_MAX != None:
            def current_constraint(model,n,m,t):
                if value(model.y_branch_magsqr[n,m]) != 0:
                    return model.current_sqr[n,m,t] <= (self.I_MAX/self.I2_BASE)**2
                else:
                    return Constraint.Skip
            model.C12 = Constraint(model.n, model.m, model.t, rule=current_constraint)
            
        
        return model
    
    def DCOPF(self):
        
        model = ConcreteModel()    
        
        model.n = Set(initialize = self.nodes)
        model.m = Set(initialize = model.n)
        model.t = RangeSet(self.st, self.ft, doc= 'periods/timesteps')

        model.line_R = Param(model.n, model.m, initialize = self.R1, default=0)
        model.line_X = Param(model.n, model.m, initialize = self.X1, default=0)
        
        '''Constructing the admittance matrix'''
        def branch_series_admittance(model,n,m):
            if (model.line_R[n,m]**2+model.line_X[n,m]**2) !=0:
                return complex((model.line_R[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)),\
                           -(model.line_X[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)))
            else:
                return 0
        model.y_series = Param(model.n,model.m, initialize = branch_series_admittance)
        
        y_series_mags_sqr = {}
        for k, v in model.y_series.items():
            if v != 0:
                y_series_mags_sqr[k] = v.real**2 + v.imag**2
            else:
                y_series_mags_sqr[k] = 0
        model.y_branch_magsqr = Param(model.n, model.m, initialize = y_series_mags_sqr)
        # model.y_branch_magsqr.pprint()
        
        Y = {}
        for k,v in model.y_series.items():
            Y[k] = v
        #print(Y)
        
        diags = {n:(n,n) for n in self.nodes}
        non_diags = {(n,k):0 for n in self.nodes for k in self.nodes if n!=k}
        # print(non_diags)
        
        ndiags = []
        for i in self.trf_keys:
            ndiags.append(i)
            ndiags.append((i[1], i[0]))
        ndiags.extend(self.lines)
        reverse_lines = [(sub[1], sub[0]) for sub in self.lines]
        ndiags.extend(reverse_lines)
        # print(ndiags)
        
        y_diags = {}
        for k, (v1,v2) in diags.items():
            # print(k)
            y_diags[k,k] = sum(v*1/(self.a[k1,k2]*np.conj(self.a[k1,k2])) for (k1,k2),v in Y.items() if k==k1 or k==k2)
        #print("DIAGONALS")
        # print(y_diags)
        
        
        y_nd = non_diags
        for key in ndiags:
        # for key,v in non_diags.items():
            y_nd[key[0],key[1]] = - sum(v*1/np.conj(self.a[k1,k2]) for (k1,k2),v in Y.items() if key[0]==k1 and key[1]==k2) - \
                sum(v*1/self.a[k1,k2] for (k1,k2),v in Y.items() if key[0]==k2 and key[1]==k1)
        #print("NON-DIAGONALS")
        # print(y_nd)
            
        Admittance = {**y_diags,**y_nd}

        Conductance = {}
        Susceptance = {}
        for k, v in Admittance.items():
            Conductance[k]= v.real
            Susceptance[k]= v.imag
        
        # G = Conductance
        B = Susceptance
        
        # =============================================================================
        # Variables and Constraints
        # =============================================================================
        model.P = Var(model.n, model.t,  bounds = (None,None), doc = 'node active power', initialize = 0)
        model.theta = Var(model.n, model.t, bounds = (None,None), doc = 'voltage angle', initialize = 0)
        
        def P_balance(model,n,t):
            return model.P[n,t] == sum(B[n,m]*(model.theta[n,t]-model.theta[m,t]) for m in model.m)
        model.DC1 = Constraint(model.n,model.t, rule = P_balance)
        
        for t in model.t:
            model.theta[self.slack,t].fix(np.radians(0))
        
        def theta_upper(model,n,t):
            if n != self.slack:
                return model.theta[n,t] <= np.pi
            else:
                return model.theta[n,t] == 0
        model.DC2 = Constraint(model.n,model.t, rule = theta_upper)
        
        def theta_lower(model,n,t):
            if n != self.slack:
                return model.theta[n,t] >= (-np.pi)
            else:
                return model.theta[n,t] == 0
        model.DC3 = Constraint(model.n,model.t, rule = theta_lower)
        
        def Non_gen_P(model,n,t):
            if n !=self.slack and n not in self.gen:
                return model.P[n,t] == 0
            else:
                return Constraint.Skip
        model.DC4 = Constraint(model.n,model.t, rule = Non_gen_P)
            
        
        return model