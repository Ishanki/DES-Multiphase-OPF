# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 14:11:36 2021

@author: Ishanki
"""

from pyomo.environ import *
import pandas as pd
import numpy as np

# TODO: Add current calcs for ISC1 and ISC3 (see ISC document in literature)
# TODO: Add shunt admittances if available?
# TODO: I changed the network transformer secondary side from wye to wye-g

class UnbalancedPowerFlow(object):
    def __init__(self,
                 st,
                 ft,
                 slack,
                 slack_angles,
                 phase_shift,
                 df_source,
                 df_buses,
                 df_loads,
                 df_lines,
                 df_linecodes,
                 df_transformer,
                 primary_buses,
                 substation,
                 # df_loadsinfo,
                 # elec_house,
                 ):
        
        self.st = st
        self.ft = ft
        self.slack = slack
        self.slack_angles = slack_angles
        self.phase_shift = phase_shift
        self.df_source = df_source
        self.df_buses = df_buses
        self.df_loads = df_loads
        self.df_lines = df_lines
        self.df_linecodes = df_linecodes
        self.df_transformer = df_transformer
        self.primary_buses=primary_buses
        self.secondary_slack = substation
        # self.df_loadsinfo = df_loadsinfo
        # self.elec_house = elec_house
        
        self.nodes = [i for i in self.df_buses.Busname]
        self.lines = list(zip(self.df_lines.Bus1, self.df_lines.Bus2))
        self.phases = ['a', 'b', 'c']
        # print(self.lines)
        # connectivity = {k: self.lines[k-1] for k in range(1,len(self.lines)+1)}
        
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
        
        
        ## Building Ymatrix within the initialisation
        self.build_Ymatrix()
    
    def build_Ymatrix(self):
        R1 = {}
        X1 = {}
        R0 = {}
        X0 = {}
        for i in range(len(self.lines)):
            for line in self.df_linecodes.Name:
                if self.df_lines.LineCode[i] == line:
                    val = self.df_linecodes.index[self.df_linecodes.Name==line].values.astype(int)[0]
                    # Converting from Ohms/km to Ohms by multiplying R or X with length
                    R1[self.lines[i]] = self.df_linecodes.R1[val]*self.df_lines.Length[i]/1000
                    X1[self.lines[i]] = self.df_linecodes.X1[val]*self.df_lines.Length[i]/1000
                    R0[self.lines[i]] = self.df_linecodes.R0[val]*self.df_lines.Length[i]/1000
                    X0[self.lines[i]] = self.df_linecodes.X0[val]*self.df_lines.Length[i]/1000
        # print(R0)
        # print(X1)
        
        list_Yseries = []
        Yseries = {}
        for i in range(len(self.lines)):
            # if i < 10:
                ## Using the Sequence frame model to obtain sequence impedances
                ## These are in Ohms
                # z_seq = np.zeros((3,3), dtype=np.complex_)  # sequence impedance
                # z_seq[0,0] = complex(R0[self.lines[i]],X0[self.lines[i]])
                # z_seq[1,1] = complex(R1[self.lines[i]],X1[self.lines[i]])
                # z_seq[2,2] = complex(R1[self.lines[i]],X1[self.lines[i]])
                # y_seq = np.linalg.inv(z_seq)  # sequence admittance
        
                # d = {}
                # for r in range(len(y_seq)):
                #     for c in range(len(y_seq[r])):
                #         d[r,c] = y_seq[r,c]
                # print(d)
                # Yseries[self.lines[i][0], self.lines[i][1]] = y_seq
                # list_Yseries.append(y_seq)
                
                ## Using the Approximate Line Model to obtain phase impedances
                # z_apprx = np.zeros((3,3), dtype=np.complex_)  # approx phase impedance
                if self.lines[i][0] in self.primary_buses and self.lines[i][1] in self.primary_buses:
                    z0 = complex(R0[self.lines[i]],X0[self.lines[i]])/self.Z1_BASE
                    z1 = complex(R1[self.lines[i]],X1[self.lines[i]])/self.Z1_BASE
                else:
                    z0 = complex(R0[self.lines[i]],X0[self.lines[i]])/self.Z2_BASE
                    z1 = complex(R1[self.lines[i]],X1[self.lines[i]])/self.Z2_BASE
                dd = ((2*z1)+z0)/3  # diagonals
                offd = (z0-z1)/3  # off-diagonals
                z_apprx = np.array([[dd, offd, offd], 
                                   [offd, dd, offd],
                                   [offd, offd, dd]])
                # if i == 14:
                #     print(z_apprx)
                y_apprx = np.linalg.inv(z_apprx)  # approximate phase admittance
                Yseries[self.lines[i][0], self.lines[i][1]] = y_apprx
                list_Yseries.append(y_apprx)
                # if i == 1:
                #     print(z_apprx)
                #     print("")
                #     print(y_apprx)
               
        # print(" ")
        # print(Yseries)
        
        diags = {n:(n,n) for n in self.nodes}
        non_diags = {(n,k):0 for n in self.nodes for k in self.nodes if n!=k}
        
        # y_diags = {}
        # for k, (v1,v2) in diags.items():
        #     # print(k)
        #     y_diags[k,k] = sum(v for (k1,k2),v in Yseries.items() if k==k1 or k==k2)
        # #print("DIAGONALS")
        # # print(y_diags)
        
        # y_nd = {}
        # for (kk1,kk2),v in non_diags.items():
        #     y_nd[kk1,kk2] = - sum(v for (k1,k2),v in Yseries.items() if kk1==k1 and kk2==k2) - \
        #         sum(v for (k1,k2),v in Yseries.items() if kk1==k2 and kk2==k1)
        # #print("NON-DIAGONALS")
        # # print(y_nd)
        
        ndiags = []
        ndiags.extend(self.lines)
        reverse_lines = [(sub[1], sub[0]) for sub in self.lines]
        ndiags.extend(reverse_lines)
        # print(ndiags)
        
        y_diags = {}
        for k, (v1,v2) in diags.items():
            # print(k)
            y_diags[k,k] = sum(v for (k1,k2),v in Yseries.items() if k==k1 or k==k2)
        #print("DIAGONALS")
        #print(y_diags)
        
        
        y_nd = non_diags
        for key in ndiags:
        # for key,v in non_diags.items():
            y_nd[key[0],key[1]] = - sum(v for (k1,k2),v in Yseries.items() if key[0]==k1 and key[1]==k2) - \
                sum(v for (k1,k2),v in Yseries.items() if key[0]==k2 and key[1]==k1)
        #print("NON-DIAGONALS")
        # print(y_nd)
        
        admittance = {**y_diags,**y_nd}
        # print('******')
        # print(admittance[0,0])
        # print('******')
        
        ## Phase shift is already considered in the transformer model used
        for trf in range(len(self.df_transformer)):
            # Ztseries = (0.00086528+0.0086528j)
            ## Transformer impedance in p.u.
            Ztseries = complex(self.df_transformer['% resistance'][trf]/1e2,
                                self.df_transformer['%XHL'][trf]/1e2)
            print(Ztseries)
            P_bus = self.df_transformer.iloc[trf]['bus1']    # Primary bus
            S_bus = self.df_transformer.iloc[trf]['bus2']    # Secondary bus
            P_conn = self.df_transformer.iloc[trf]['Conn_pri'].lower()  # Primary connection
            S_conn = self.df_transformer.iloc[trf]['Conn_sec'].lower()  # Secondary connection
            # print(P_bus, S_bus)
            y_series = 1/Ztseries
            ## Y2 appears in the self admittances. To prevent singularities, a small shunt admittance is added
            Ysmall = 1e-7*np.eye(3)*np.abs(np.real(y_series)) 
            # print(Ysmall)
            y_shunt = 0
            Yshunt = y_shunt*np.eye(3)
            
            if P_conn =='delta' and S_conn=='wye-g':
                ## Y series for this has to be in per unit!
                ## Matrices from Chen
                # Y_pp = 1/3*np.array([[2*y_series,-y_series,-y_series],
                #                       [-y_series,2*y_series,-y_series],
                #                       [-y_series,-y_series,2*y_series]]) \
                #         + Ysmall
                # Y_ss = np.array([[y_series,0,0],\
                #                   [0,y_series,0],\
                #                   [0,0,y_series]]) + Ysmall
                # Y_sp = 1/np.sqrt(3)*np.array([[-y_series,y_series,0],
                #                                 [0,-y_series,y_series],
                #                                 [y_series,0,-y_series]])\
                #         - Ysmall
                # Y_ps = Y_sp.T
                
                ##Matrices from OPEN
                Y_pp = 1/3*np.array([[2*y_series,-y_series,-y_series],
                                      [-y_series,2*y_series,-y_series],
                                      [-y_series,-y_series,2*y_series]])+ Ysmall
                Y_ss = np.array([[y_series,0,0],\
                                  [0,y_series,0],\
                                  [0,0,y_series]]) + Ysmall
                Y_ps = (1/np.sqrt(3))*np.array([[-y_series,y_series,0],
                                                [0,-y_series,y_series],
                                                [y_series,0,-y_series]])- Ysmall

                Y_sp = Y_ps.T
            
            elif P_conn =='wye-g' and S_conn=='wye-g':
                Y_pp = Ysmall + np.array([[y_series,0,0],\
                                  [0,y_series,0],\
                                  [0,0,y_series]])
                Y_ss = Ysmall + np.array([[y_series,0,0],\
                                  [0,y_series,0],\
                                  [0,0,y_series]])
                Y_sp = -Ysmall -np.array([[y_series,0,0],\
                                  [0,y_series,0],\
                                  [0,0,y_series]])
                Y_ps = -Ysmall -np.array([[y_series,0,0],\
                                  [0,y_series,0],\
                                  [0,0,y_series]])
            
            else:
                print("No transformer model available yet")
            
            ## TODO: Add if-else for transformer turns ratio depending on the different types of transformers
            # if P_conn != S_conn:
            # if self.V1_BASE != self.V2_BASE:
            #     Y_pp = Y_pp*1/(self.TRF_RATIO**2)
            #     Y_sp = Y_sp*1/self.TRF_RATIO
            #     Y_ps = Y_ps*1/self.TRF_RATIO
                
                # print("got here...")
                # alpha = self.PU_SOURCE/1.0
                # Y_pp = Y_pp*1/(alpha**2)
                # Y_sp = Y_sp*1/alpha
                # Y_ps = Y_ps*1/alpha
                # print(Y_pp)
                # print("*"*10)
                
                # a_trf = self.TRF_RATIO*np.exp(1j*np.radians(30))
                # print(a_trf)
                # astar_trf = np.conj(a_trf)
                # A_i = a_trf*np.eye(3) #self.TRF_RATIO*np.eye(3)
                # # ax = a_trf
                # # ax_star = astar_trf#*np.eye(3)
                
                # # Y_pp = Y_pp*1/(ax_star*ax)
                # # Y_sp = Y_sp*1/ax
                # # Y_ps = Y_ps*1/ax_star
                
                # Y_pp = np.matmul(np.matmul(A_i,Y_pp),A_i.T)
                # Y_ps = np.matmul(A_i,Y_ps)
                # Y_sp = np.matmul(Y_sp,A_i.T)
            
            ## This is from the OPEN code for testing
            if 'kV_pri' in self.df_transformer.columns and \
                'kV_sec' in self.df_transformer.columns:
                A_i = np.eye(3)/(self.df_transformer.iloc[trf]['kV_pri']
                            /self.df_transformer.iloc[trf]['kV_sec'])
                # print(A_i)
            else:
                A_i = np.eye(3)
            Y_pp = np.matmul(np.matmul(A_i,Y_pp),A_i.T)
            Y_ps = np.matmul(A_i,Y_ps)
            Y_ss = Y_ss
            Y_sp = np.matmul(Y_sp,A_i.T)
            
            # print(f"{Y_pp}\n\n{Y_ss}\n\n{Y_ps}\n\n{Y_sp}")
            # print('________________________')
            # print(admittance[0,1])
            # print('________________________')
            
            admittance[P_bus,P_bus] = admittance[P_bus,P_bus]+0.5*Yshunt+Y_pp
            admittance[S_bus,S_bus] = admittance[S_bus,S_bus]+0.5*Yshunt+Y_ss
            admittance[S_bus,P_bus] = Y_sp
            admittance[P_bus,S_bus] = Y_ps
        
        # for (k1, k2), v in admittance.items():
        #     if k1 == self.slack and k2 == self.slack:
        #         R1=0.0000001
        #         X1=0.0000001
        #         R0=0.0000001
        #         X0=0.0000001
        #         z0 = complex(R0,X0)/self.Z2_BASE     
        #         z1 = complex(R1,X1)/self.Z2_BASE
        #         dd = ((2*z1)+z0)/3  # diagonals
        #         offd = (z0-z1)/3  # off-diagonals
        #         source_z = np.array([[dd, offd, offd], 
        #                            [offd, dd, offd],
        #                            [offd, offd, dd]])
        #         source_y = np.linalg.inv(source_z)
        #         print(source_y)
        #         admittance[k1,k2] = admittance[k1,k2]+source_y
                
        # print('________________________')
        # print(admittance[0,0])
        # print('________________________\n')
        # # print(admittance[0,1])
        # # print('________________________')
        # print(f"{admittance[2,1]}\n\n{admittance[2,2]}\n\n{admittance[2,3]}\n\n")
        # print('________________________')
        
        self.Y_dict = {}
        for (k1, k2), v in admittance.items():
            if type(v) is np.ndarray:
                self.Y_dict[k1, k2, 'a', 'a'] = v[0,0]
                self.Y_dict[k1, k2, 'a', 'b'] = v[0,1]
                self.Y_dict[k1, k2, 'a', 'c'] = v[0,2]
                self.Y_dict[k1, k2, 'b', 'a'] = v[1,0]
                self.Y_dict[k1, k2, 'b', 'b'] = v[1,1]
                self.Y_dict[k1, k2, 'b', 'c'] = v[1,2]
                self.Y_dict[k1, k2, 'c', 'a'] = v[2,0]
                self.Y_dict[k1, k2, 'c', 'b'] = v[2,1]
                self.Y_dict[k1, k2, 'c', 'c'] = v[2,2]
            else:
                for p1 in self.phases:
                    for p2 in self.phases: 
                        self.Y_dict[k1, k2, p1, p2] = complex(0, 0)
        
        
        # self.Y_dictmod = pd.DataFrame()
        # df_index = []
        # for n in range(0,907):
        #     for p in self.phases:
        #         key = [n,p]
        #         df_index.append(key)
        
        # print(self.Y_dict)
        # resY = pd.DataFrame.from_dict(self.Y_dict, orient="index")
        # resY.to_excel("res_Yx.xlsx")
        
    def UPF(self, power_init, load_init):
        gen = [i for i in self.df_loads.Bus]
        # print(gen)
    
        Conductance = {}
        Susceptance = {}
        for k, v in self.Y_dict.items():
            Conductance[k]= v.real
            Susceptance[k]= v.imag
        
        ## Creating Pyomo model and adding OPF constraints
        model = ConcreteModel()
        model.n = Set(initialize = self.nodes)
        model.m = Set(initialize = model.n)
        model.t = RangeSet(self.st, self.ft, doc= 'periods/timesteps')
        model.p = Set(initialize=self.phases)
        
        G = Conductance
        B = Susceptance
        
        V_init = {}
        for n in self.nodes:
            for p in self.phases: 
                for t in model.t:
                    if n==self.slack:
                        ## TODO: Better initialisation required here
                        # V_init[n,p,t]=230*1.1
                        V_init[n,p,t] = self.PU_SOURCE
                    else:
                        # V_init[n,p,t]=230*1.1
                        V_init[n,p,t] = self.PU_SOURCE
                
        theta_init = {}
        for n in self.nodes:
            for p in self.phases: 
                for t in model.t:
                    if n in self.primary_buses:
                        theta_init[n,'a',t] = np.radians(self.slack_angles[0])
                        theta_init[n,'b',t] = np.radians(self.slack_angles[1])
                        theta_init[n,'c',t] = np.radians(self.slack_angles[2])
                    else:
                        theta_init[n,'a',t] = np.radians(self.slack_angles[0]+\
                                                         self.phase_shift)
                        theta_init[n,'b',t] = np.radians(self.slack_angles[1]+\
                                                         self.phase_shift)
                        theta_init[n,'c',t] = np.radians(self.slack_angles[2]+\
                                                         self.phase_shift)
            
        load_phase = dict(zip(self.df_loads.Bus, self.df_loads.phases))
        # P_loads = self.df_loadsinfo.set_index(['Bus', 'Time'])['Load'].to_dict()
        
        P_init = {(n,p,t):0 for n in self.nodes for p in self.phases \
                  for t in model.t}
        for load, phase in load_phase.items():
            for (bus,time), val in power_init.items():
                if load == bus:
                    P_init[bus, phase.lower(), time] = -val/self.S_BASE
        
        PF_dict = dict(zip(self.df_loads.Bus, self.df_loads.PF))
        Q_init = {(n,p,t):0 for n in self.nodes for p in self.phases for t in model.t}
        for load, phase in load_phase.items():
            for (bus,time), v in load_init.items():
                if load == bus:
                    if v!=0:
                        Q_init[bus, phase.lower(), time] = \
                            -np.sqrt((v**2)*((1/(PF_dict[load]**2))-1))\
                                /self.S_BASE
        
        for n in self.nodes:
            for p in self.phases:
                for t in model.t:
                    if n==self.secondary_slack:
                        P_init[n,p,t] = sum(P_init[n,p,t] for n in self.nodes \
                                            if n!=self.slack)*-1
                        Q_init[n,p,t] = sum(Q_init[n,p,t] for n in self.nodes \
                                            if n!=self.slack)*-1
        
        with open('init_check.txt','w') as f:
            for k, v in P_init.items():
                f.write(f'{k}: {v}\n')
            f.write('\n\n')
            for k, v in Q_init.items():
                f.write(f'{k}: {v}\n')
            f.write('\n\n')
            for k, v in V_init.items():
                f.write(f'{k}: {v}\n')
            f.write('\n\n')
            for k, v in theta_init.items():
                f.write(f'{k}: {v}\n')
            f.write('\n')
        
        model.V = Var(model.n, model.p, model.t, bounds = (None,None), 
                      doc = 'node voltage', initialize = V_init)
        model.P = Var(model.n, model.p, model.t,  bounds = (None,1e7), 
                      doc = 'node active power', initialize = P_init)
        model.Q = Var(model.n, model.p, model.t,  bounds = (None,1e7), 
                      doc = 'node reactive power', initialize = Q_init)
        model.theta = Var(model.n, model.p, model.t, bounds = (-np.pi,np.pi), 
                          doc = 'voltage angle', initialize = theta_init)
        model.dummy = Var(model.n, model.p, model.t, bounds = (None,None), 
                          doc = 'dummy var to give DoF to IPOPT', initialize = 0)
        
        ## This is a dummy constraint to give an extra DoF to allow IPOPT to solve
        def dummyc(model, t):
            return sum(model.dummy[n,p,t] for n in model.n for p in model.p) >= 1
        model.DC1 = Constraint(model.t, rule=dummyc)
        
        # def slack_power_constraint1(model,p,t):
        #     return model.P[self.secondary_slack,p,t] >= sum(model.P[n,p,t]for n in self.nodes if n!=self.secondary_slack)*-1
        # model.C0a = Constraint(model.p, model.t, rule = slack_power_constraint1)
        
        # def slack_power_constraint2(model,p,t):
        #     return model.Q[self.secondary_slack,p,t] >= sum(model.Q[n,p,t] for n in self.nodes if n!=self.secondary_slack)*-1
        # model.C0b = Constraint(model.p, model.t, rule = slack_power_constraint2)
        
        def Pa_balance(model,n,t):
            return model.P[n,'a',t] == model.V[n,'a',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'a',p]*cos(model.theta[n,'a',t]-model.theta[m,p,t]))\
                                    + (B[n,m,'a',p]*sin(model.theta[n,'a',t]-model.theta[m,p,t])))for m in model.m) for p in model.p))
        model.C1a = Constraint(model.n, model.t, rule = Pa_balance)
        
        def Pb_balance(model,n,t):
            return model.P[n,'b',t] == model.V[n,'b',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'b',p]*cos(model.theta[n,'b',t]-model.theta[m,p,t]))\
                                    + (B[n,m,'b',p]*sin(model.theta[n,'b',t]-model.theta[m,p,t])))for m in model.m) for p in model.p))                                              
        model.C1b = Constraint(model.n, model.t, rule = Pb_balance)
        
        def Pc_balance(model,n,t):
            return model.P[n,'c',t] == model.V[n,'c',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'c',p]*cos(model.theta[n,'c',t]-model.theta[m,p,t]))\
                                    + (B[n,m,'c',p]*sin(model.theta[n,'c',t]-model.theta[m,p,t])))for m in model.m) for p in model.p))
        model.C1c = Constraint(model.n, model.t, rule = Pc_balance)
        
        def Qa_balance(model,n,t):
            return model.Q[n,'a',t] ==  model.V[n,'a',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'a',p]*sin(model.theta[n,'a',t]-model.theta[m,p,t]))\
                                    - (B[n,m,'a',p]*cos(model.theta[n,'a',t]-model.theta[m,p,t]))) for m in model.m) for p in model.p))
        model.C2a = Constraint(model.n,model.t, rule = Qa_balance)
        
        def Qb_balance(model,n,t):
            return model.Q[n,'b',t] ==  model.V[n,'b',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'b',p]*sin(model.theta[n,'b',t]-model.theta[m,p,t]))\
                                    - (B[n,m,'b',p]*cos(model.theta[n,'b',t]-model.theta[m,p,t]))) for m in model.m) for p in model.p))
        model.C2b = Constraint(model.n,model.t, rule = Qb_balance)
        
        def Qc_balance(model,n,t):
            return model.Q[n,'c',t] ==  model.V[n,'c',t]*(sum(sum(model.V[m,p,t]*((G[n,m,'c',p]*sin(model.theta[n,'c',t]-model.theta[m,p,t]))\
                                    - (B[n,m,'c',p]*cos(model.theta[n,'c',t]-model.theta[m,p,t]))) for m in model.m) for p in model.p))
        model.C2c = Constraint(model.n,model.t, rule = Qc_balance)
        
        def V_upper(model,n,p,t):
            if n == self.slack:
            #     ## Primary side slack bus voltage fixed 
            #     # return Constraint.Skip
                return model.V[n,p,t] == self.PU_SOURCE
            else:
                # Secondary side bus voltage upper bound
                return Constraint.Skip
                # return model.V[n,p,t] <= self.V_UB/self.V2_BASE
        model.C3 = Constraint(model.n,model.p,model.t, rule = V_upper)
        
        def V_lower(model,n,p,t):
            if n == self.slack:
            #     ## Primary side slack bus voltage fixed 
            #     # return Constraint.Skip
                return model.V[n,p,t] == self.PU_SOURCE
            else:
                ## Secondary side bus voltage lower bound
                # return Constraint.Skip
                return model.V[n,p,t] >= self.V_LB/self.V2_BASE
        model.C4 = Constraint(model.n,model.p,model.t, rule=V_lower)
        
        # print(self.V_UB/self.V2_BASE)
        # print(self.V_LB/self.V2_BASE)
        
        # ANGLE_FLEX = 10
        # ##TODO: Angle limits 
        # def angles_lower(model,n,p,t):
        #     if n != self.slack:
        #         if p=='a':
        #             return model.theta[n,'a',t] >= np.radians(self.slack_angles[0]+self.phase_shift-ANGLE_FLEX)
        #         elif p=='b':
        #             return model.theta[n,'b',t] >= np.radians(self.slack_angles[1]+self.phase_shift-ANGLE_FLEX)
        #         else:
        #             return model.theta[n,'c',t] >= np.radians(self.slack_angles[2]+self.phase_shift-ANGLE_FLEX)
        #     else:
        #         return Constraint.Skip
        # model.C5 = Constraint(model.n,model.p,model.t, rule=angles_lower)
        
        # def angles_upper(model,n,p,t):
        #     if n != self.slack:
        #         if p=='a':
        #             return model.theta[n,'a',t] <= np.radians(self.slack_angles[0]+self.phase_shift+ANGLE_FLEX)
        #         elif p=='b':
        #             return model.theta[n,'b',t] <= np.radians(self.slack_angles[1]+self.phase_shift+ANGLE_FLEX)
        #         else:
        #             return model.theta[n,'c',t] <= np.radians(self.slack_angles[2]+self.phase_shift+ANGLE_FLEX)
        #     else:
        #         return Constraint.Skip
        # model.C6 = Constraint(model.n,model.p,model.t, rule=angles_upper)
        
        for t in model.t:
            model.theta[self.slack,'a',t].fix(np.radians(self.slack_angles[0]))
            model.theta[self.slack,'b',t].fix(np.radians(self.slack_angles[1]))
            model.theta[self.slack,'c',t].fix(np.radians(self.slack_angles[2]))
        
        ## Checking for generational/load nodes and setting power injections to zero
        def Non_gen_P(model,n,p,t):
            if n !=self.secondary_slack and n not in gen:
                return model.P[n,p,t] == 0
            else:
                return Constraint.Skip
        model.C7 = Constraint(model.n,model.p,model.t, rule = Non_gen_P)
        
        def Non_gen_Q(model,n,p,t):
            if n != self.secondary_slack and n not in gen:
                return model.Q[n,p,t] == 0
            else:
                return Constraint.Skip
        model.C8 = Constraint(model.n,model.p,model.t, rule = Non_gen_Q)
        
        # TODO: Current limits
        # TODO: VUB calculation
        
        # TODO: In C3 and C4, transfer those params to the network file
                
        return model

        
        
    

