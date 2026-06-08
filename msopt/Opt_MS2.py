"""
Module for gradient ascent optimization
Last update: 2025/04/23 by munseong
"""
import numpy as np
import autograd.numpy as npa
from autograd import tensor_jacobian_product
import os

from . import Sub_Mapping


"""
Optimization methods for various cases

O1. Goal_Attainment: for multi-objective optimization -> use 1st order momentum gradient
O2. Minimax: for multi-objective optimization -> use minimum gradients only 
O3. Momentum: Adaptive momentum estimation (Adam) -> for escape from saddle point and bias correction

O4. Back_tracking_init: Define the parameters for backtracking
O5. Back_tracking: evaluate the current learning rate with iterative process
O6. Back_tracking_call: extract the single set of data
O7. Back_traking_record: save all evaluation results in the list
"""
""" O1 """
def Goal_Attainment(dT_old, fom, dJ_du,
): # Method for multiobjective optimization
    T_mean= np.mean(fom)
    dT_cur= dT_old*0
    for i in range (0, len(fom)): # for n th object
        print(fom[i])
        if fom[i] - T_mean <= 0: # constrain
            dT_cur += dJ_du[i]#/(npa.clip(npa.max(npa.abs(dJ_du[i])), 1e-9, npa.inf)) # + normaized dj_du
        else:
            dT_cur += dT_old # + dt_du
    dT_cur=dT_cur/len(fom) # dt_du
    return dT_cur

""" O2 """
def Minimax(fom, dJ_du,
): # Method for multiobjective optimization
    T_mean= np.mean(fom)
    dT_cur= dJ_du[0]*0
    Weight_sum= 0
    for i in range (0, len(fom)): # for n th object
        print(fom[i])
        #Weight= 1-fom[i]
        if fom[i] - T_mean <= 0: # constrain
            dT_cur += dJ_du[i]#/(npa.clip(npa.max(npa.abs(dJ_du[i])), 1e-9, npa.inf)) # + normaized dj_du
            Weight_sum +=1
    dT_cur=dT_cur/Weight_sum # dt_du
    return dT_cur

""" O3 """
def Momentum(dF_cur, dF_old, dF_old2, numevl
): # Momentum (Adam)
    N_grad = 9
    bt1= N_grad/10
    grad_adj = bt1*dF_old + (1-bt1)*dF_cur
    RMSprop = (1-(1-bt1)**2)*(dF_old2) + ((1-bt1)**2)*(dF_cur**2)
    Bias_corr = RMSprop/(1-bt1**(numevl+1))
    grad_prop = grad_adj/ (np.sqrt(Bias_corr) + 1e-8)
    return grad_prop, grad_adj, RMSprop

""" O4 """
def Back_tracking_init(f0, f0s, bt_tol):
    foms_info=[0]*bt_tol # maximum 7 steps for single dir.
    fom_info=[0]*bt_tol    
    foms_info[0]= f0s
    fom_info[0]=f0
    BT_history=[f0]*bt_tol
    return foms_info, fom_info, BT_history, False, False

""" O5 """
def Back_tracking(fom_info, Backtraking_count, trig, is_conv):
    print(f"\n Backtracking: {Backtraking_count}/{len(fom_info)}")
    # print(f"Fom without backtracking: {fom_info[0]}")
    # print(f"Current fom: {fom_info[Backtraking_count]}")
    if fom_info[0] > fom_info[Backtraking_count]:
        print("Current fom < Initial fom")
        if trig: # alpha reset
            return True, True
        else:
            if is_conv:
                if Backtraking_count > 2:
                    if fom_info[Backtraking_count-1] >= fom_info[Backtraking_count]:
                        if fom_info[Backtraking_count-1] > fom_info[0]:
                            # print(f"Last fom is best: {fom_info[Backtraking_count-1]}")
                            return False, True
                        else: # alpha reset
                            return True, True
                    else: # Trig on
                        return True, False
                else: # Keep going
                    return False, False
            else:
                if fom_info[Backtraking_count-1] > fom_info[0]:
                    # print(f"Last fom is best: {fom_info[Backtraking_count-1]}")
                    return False, True
                else: # Trig on
                    return True, False
    else:
        print("Current fom >= Initial fom")
        if fom_info[Backtraking_count-1] >= fom_info[Backtraking_count]:
            # print(f"Last fom is best: {fom_info[Backtraking_count-1]}")
            return False, True
        else: # Keep going
            # print(f"Current fom is best")
            return False, False

""" O6 """        
def Back_tracking_call(fom_info, foms_info, Backtraking_count):
    f0=fom_info[Backtraking_count]      
    f0s=foms_info[Backtraking_count]     
    return f0, f0s

""" O7 """
def Back_traking_record(BT_history, fom_info, Backtraking_count):
    BT_history[Backtraking_count-1]=np.real(fom_info[Backtraking_count])
    return BT_history


def _finite_fom_value(fom):
    try:
        value = float(np.real(fom))
    except (TypeError, ValueError, OverflowError):
        return -np.inf
    if not np.isfinite(value):
        return -np.inf
    return value


def _is_failed_fom(fom):
    return _finite_fom_value(fom) <= -1e20

"""
P1. inv_tanh_proj: inverse function of tanh
P2. projection_error: projection error calculation for gradual evolve of the beta 
"""
""" P1 """
def inv_tanh_proj(beta, eta):
    return (npa.tanh(beta * eta) + np.log(np.abs(npa.cosh(beta)))/beta) / (npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))

""" P2 """
def projection_error(beta, eta):
    if beta>50:
        return 0.5
    A= inv_tanh_proj(beta, eta)
    for db in range(5000, 0, -1):
        dB=db/10000
        B= inv_tanh_proj(beta+dB, eta)
        Err= np.abs((B-A)/A)
        if Err < 0.004:
            break
    if dB==0.5:
        for db in range(10000, 20000, +1):
            dB=db/20000
            B= inv_tanh_proj(beta+dB, eta)
            Err= np.abs((B-A)/A)
            if Err > 0.004:
                break  
    return dB

""" P3 """
def Born_validity(dJ_dus, N_fom, top_k=50):
    for i in range(0, N_fom):
        dJ_dus[i]= npa.where(npa.isfinite(dJ_dus[i]), dJ_dus[i], 0)
        outlier_th= npa.percentile(npa.abs(dJ_dus[i]), 99.9)
        outlier_th= npa.where(npa.isfinite(outlier_th), outlier_th, 0)
        dJ_dus[i]= npa.where(npa.abs(dJ_dus[i])> outlier_th, 0, dJ_dus[i])
        born_th= npa.percentile(npa.abs(dJ_dus[i]), 100-top_k)
        born_th= npa.where(npa.isfinite(born_th), born_th, 0)
        dJ_dus[i]= npa.where(npa.abs(dJ_dus[i])>= born_th, dJ_dus[i], 0)
    return dJ_dus
        

"""
gradient ascent optimizer class

5 Outer loop functions: Updater, Warm_restarter, Conv_tol, Conv_check, After_Conv
Updater -> Warm_restarter, After_Conv
Conv_tol -> Conv_check

& 3 Inner loop functions: Design_update, Conversion, Inner_iter
"""
class OPT_Ms:
    def __init__(
        self, 
        Initial_geo, 
        Initial_grad,
        design_dir= "./A/",
        local_best_dir= "./Local_bests/",
        Load: bool = False, 
        Load_iter=0,
        Born_k=50,
        Initial_LR=0.2,
        Raw: bool = False,
    ):
        self.Array=[0]*5 #--------------|
        self.Array[0]= Initial_geo      # 0: Geometry
        self.Array[1]= Initial_geo*0    # 1: Jacobian gradient 
        self.Array[2]= Initial_grad     # 2: 1st order momentum gradient  
        self.Array[3]= Initial_grad     # 3: 2nd order momentum gradient
        self.Array[4]= 1.0              # 4: Beta for mapping function
        #
        self.Parameters=[0]*5 #---------|
        self.Parameters[0]=Initial_LR   # 0: Learning rate
        self.Parameters[1]=0            # 1: FoM
        self.Parameters[2]=0            # 2: Global best FoM
        self.Parameters[3]=0            # 3: Warm restart counter
        self.Parameters[4]=0            # 4: Convergence counter
        #
        self.Best=[0]*8 #---------------|
        self.Best[0]= 0                 # 0: Local best FoM
        self.Best[1]= Initial_geo       # 1: Local best geometry
        self.Best[2]= Initial_grad      # 2: Local best 1st order grad
        self.Best[3]= Initial_grad      # 3: Local best 2nd order grad
        self.Best[4]= self.Array[4]     # 4: Local best beta
        self.Best[5]= self.Parameters[0]# 5: Local best LR
        self.Best[6]= 0                 # 6: Local best grad stack
        self.Best[7]= 0                 # 7: Local best iters
        #
        self.Best2=[0]*3 #--------------|
        self.Best2[0]= 0                # 0: cur beta
        self.Best2[1]= 0                # 1: cur grad stack
        self.Best2[2]= 0                # 2: cur iters
        #
        self.Outer_M=[0]*3 #------------|
        self.Outer_M[0]= False          # 0: Conversion triger
        self.Outer_M[1]= 0              # 1: Conversion counter
        self.Outer_M[2]= 0              # 2: Binarization ratio
        #
        self.beta_scale= 0              # Projection sharpness scale (+100%)
        self.bt_tol= 6                  # Maximum Backtraking counts
        self.bt_max_cnt= 0              # Full backtraking triger counts
        #---------------------------------------|
        self.F_history=[0, 0, 0, 0, 0, 0, 0, 0] # FoM history for 7 iters
        self.P_history=[0, 0, 0, 0, 0]          # Penalization history
        self.cur_iter = [0]                     # Current iteration idx
        self.Born_k= Born_k                     # Threshold for gradient
        #---------------------------------------|
        self.flag= False          # True: step conversion  / False: End condition
        self.numevl= 1            # momentum stack
        self.is_converged= False  # current convergence condition
        self.Re_roll= False       # Re-scale triger of the beta step (large conversion)
        #-------------------------|
        self.local_best_dir= local_best_dir 
        self.design_dir= design_dir
        self.Raw= Raw
        self.Initial_LR=Initial_LR
        if Load:
            os.chdir(self.local_best_dir)
            Parameter=np.loadtxt(f"param_{Load_iter}.txt")
            self.Array[0]= np.loadtxt(f"ref_layer_{Load_iter}.txt")
            self.Array[4] = Parameter[0]
            self.Parameters[0] = Parameter[1]
            os.chdir("..")
        #
        self.grad_max_history = []
        self.grad_mean_history = []
        self.beta_history = []
        self.binarization_history = [] 
        self.learning_rate_history = []
        self.evaluation_history = [0]
        self.wrong_evaluation_history = []
        self.wrong_evaluation_history2 = [] 
        for i in range(self.bt_tol):
            self.wrong_evaluation_history.append([0])
            self.wrong_evaluation_history2.append([0])

    """ Outer loop 0 """
    def Updater(self, is_Best= False, is_Worst = False):
        if is_Best:                         # Save current
            self.Best[0]= self.Parameters[1]  # 1. FoM
            self.Best[1]= self.Array[0]       # 2. geometry
            self.Best[2]= self.Array[2]       # 3. 1st-order momentum
            self.Best[3]= self.Array[3]       # 4. 2nd-order momentum
            self.Best[4]= self.Best2[0]       # 5. beta
            self.Best[5]= self.Parameters[0]  # 6. learning rate
            if self.Best[0]> self.Best[6]:
                self.Best[6]= self.Best[0]   # 7. global best
            self.Best[7]= self.Best2[2]       # 8. iteration
        elif is_Worst:                      # Load best & Reset
            self.Array[0] = self.Best[1]      # 1. geometry
            self.Array[1] = self.Best[1]*0    # 2. gradient
            self.Array[2] = self.Best[2]*0    # 3. 1st-order momentum
            self.Array[3] = self.Best[3]*0    # 4. 2nd-order momentum
            self.Array[4] = self.Best[4]      # 5. beta
            self.Parameters[0] = 0.1          # 6. New LR
            self.Parameters[2] = self.Best[0] # 7. update global best FoM
            self.Parameters[3] = 0            # 8. WR counter
        else:                               # Update 
            self.Best2[0]= self.Array[4]      # 1. beta
            self.Best2[1]= self.numevl        # 2. momentum stack
            self.Best2[2]= self.cur_iter[0]+1 # 3. iterations

    """ Outer loop 1 """
    def Warm_restarter(self, Save_small: bool = True):
        if self.Parameters[1] < self.Best[0]*1.01: # 1% improvement?
            if self.Parameters[1] > self.Best[0] and Save_small: 
                self.Updater(is_Best=True) # save small improvement < 1%
                self.Parameters[3] -=1
            self.Parameters[3] +=1         # warm restart count
            if self.Parameters[3] == 3:    # warm restarter tol : 3 -> save local best info.
                if self.Parameters[0] > 0.0001:
                    self.Parameters[0] *= 5
                else:
                    self.Parameters[0] *= 20
                self.Parameters[3]= 0
                print('\n Warm restart with LR: 0.2 \n')
        else:
            self.Updater(is_Best=True)
            self.Parameters[3]= 0


    """ Outer loop 2 """
    def Conv_tol(self):
        F=self.Parameters[1]
        # Update FoM history list
        for i in range (0, len(self.F_history)-1):
            self.F_history[i]=self.F_history[i+1]
        self.F_history[len(self.F_history)-1]=F
        mean_F= npa.mean(self.F_history)
        func_tol= abs((self.F_history[len(self.F_history)-2]- F)/(mean_F + 1e-8))
        self.Parameters[4] +=1
        # evaluate the convergence condition
        if self.Parameters[4] < 5:
            self.is_converged = False
            # elif F < 0.6*npa.max(self.F_history):
            # self.is_converged = True
        elif self.Outer_M[2]> self.binarization_history[-2]*1.1:
            self.is_converged = True
        elif F > mean_F*1.001: # > 1e-5%    #################
            if func_tol < 1e-5:######
                self.is_converged = True
            else:
                self.is_converged = False
        else:
            if self.Is_gray and func_tol > 1e-7:
                self.is_converged = False
            else:
                self.is_converged = True

    """ Outer loop 3 """
    def Conv_check(self):
        if not self.flag and self.Parameters[3]==2:
            self.flag = True
            self.numevl= 1
            self.Updater(is_Worst= True)
        self.Conv_tol()
        if self.is_converged:
            self.F_history=[0, 0, 0, 0, 0, 0, 0]
            np.savetxt(f"{self.local_best_dir}ref_layer_{self.Best[7]}.txt", self.Best[1])
            np.savetxt(f"{self.local_best_dir}param_{self.Best[7]}.txt", [self.Best[4], self.Best[5], self.Best[0]])
            self.Updater(is_Worst= True)
            if self.Outer_M[2] >= 0.5 or self.Outer_M[2] < 0.05:
                self.beta_scale = 0.5 # 150%
            else:
                self.beta_scale = self.Outer_M[2]*1.5 # 200%
            self.Array[4] *= (1.0 + self.beta_scale)
            self.P_history= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            if self.Best[4] > 25 and self.Outer_M[2] >= 0.95 or self.Is_gray: 
                # beta > 30 (stable projection) + binarizaton > 95%
                self.Updater(is_Worst= True)
                self.Parameters[2] = -777   # exit condition
                self.is_converged = False   # skip the AC routine
                self.Parameters[0] = 0
            print('\n------- FoM converged ------\n')
            self.flag= True
            self.numevl= 0

    """ Outer loop 4 """
    def After_Conv(self):
        self.Parameters[4]=0  # Convergen tol
        if self.Outer_M[1] == 10: # continue with current beta
            self.Outer_M[0] = False
            self.Outer_M[1] = 0
            self.Re_roll= False
            self.Parameters[0] = self.Initial_LR
            self.Best[0] = 0 # Reset local best FoM
        else:
            self.P_history[self.Outer_M[1]]=self.Parameters[1]
            max_step= self.P_history.index(max(self.P_history))
            # Last beta step > current step
            if self.Outer_M[1] ==5 and self.Parameters[1] < self.P_history[max_step]:
                self.beta_scale *= 10**(self.Outer_M[1]-max_step)
                self.Outer_M[1], self.numevl= 10, 0 # Exit triger
                self.Updater(is_Worst= True)
                self.Array[4] *= (1.0 + self.beta_scale)
            elif self.Parameters[1] < self.Parameters[2]*0.9:
                if self.Outer_M[1]==5:
                    self.Updater(is_Worst= True)
                    self.Outer_M[1], self.numevl= 10, 0 # Exit triger
                    self.flag=False
                else:
                    self.Outer_M[0] = True # Re-scale triger
                    self.Outer_M[1] +=1    # Re-scale counter
                    self.Re_roll= True     # Re-scale mode
                    self.numevl= 0         # reset momentum
                    self.Updater(is_Worst= True)
                    #  0     1     2      3        4        5
                    # 150-> 105, 100.5, 100.05, 100.005, 100.0005 %  
                    # 200-> 110, 101,   100.1,  100.01,  100.001% 
                    self.beta_scale *=0.1                 
                    self.Array[4] *= (1.0 + self.beta_scale)
            else:
                self.Outer_M[0] = False
                self.Outer_M[1] = 0
                self.Re_roll= False
                self.Parameters[0] = self.Initial_LR
                self.Best[0] = 0 # Reset local best FoM



    """ Inner loop """
    def Design_update(self, v, alpha, g, mapping, beta, is_init=False):
        Updated_v= npa.clip(v+ alpha*g, 0.0, 1.0)                       
        X=mapping(Updated_v, beta)
        if is_init:
            print("\n Current iteration: {}".format(self.cur_iter[0] + 1))
            if self.Re_roll:
                print(f"Beta scale: {1.0+self.beta_scale}")
                self.beta_history[-1]=beta    
            else:
                self.beta_history.append(beta)
            if alpha == 0 and self.flag:
                beta= npa.inf
                self.Array[4]=beta
                if self.Is_gray:
                    pass
                else:
                    X= Sub_Mapping.tanh_projection_m(X, beta, 0.5)
                self.Outer_M[2] = npa.sum(npa.where(((1e-3<X)&(X<(1-1e-3))), 0, 1))/X.size
                print("Current beta: ", beta)                    
                print(f"Binarization rate: {round(self.Outer_M[2]*100,2)}%")        
                if self.Re_roll:
                    self.binarization_history[-1]=self.Outer_M[2]
                else:
                    self.binarization_history.append(self.Outer_M[2])  
        return Updated_v, X

    def Design_update_AC(self, beta_n, mapping, v, X):
        X_best = mapping(v, self.Best[4])
        grad_temp = X_best - X                                      
        gJ = v*0
        # !!! this is not adjoint gradient !!!                                    
        gJ[:] = tensor_jacobian_product(mapping, 0)(v, beta_n, grad_temp)
        return gJ

    def Conversion(self, mapping, v, X):
        if self.flag:
            beta_n = self.Array[4]
            if self.numevl == 0:
                self.numevl +=1
                if self.Array[4] == self.Best[4]:
                    print("Continue with last beta \n")
                else:
                    if self.Outer_M[0]:
                        print("Beta updated with rescaled step \n")
                    else:
                        print("Beta updated with Large step \n")
                    gJ = self.Design_update_AC(beta_n, mapping, v, X)
                    grad_suff = 0 
                    v_new, X = self.Design_update(v, self.Parameters[0], gJ, mapping, beta_n)
                    return v_new, X, gJ, grad_suff, beta_n 
            else:
                print("Continue with fixed beta \n")    
        else:
            del_beta= projection_error(self.Array[4], 0.05)
            beta_n = self.Array[4] + del_beta
            print("Beta upated with gradual step \n")
        X0=mapping(npa.clip(v, 0.0, 1.0), beta_n)
        go=self.Array[2]
        grad_suff=npa.sum(npa.abs(npa.where(((X0==0)&(go<0)),0, npa.where(((X0==1)&(go>0)),0,go))))        
        return v, X, self.Array[1], grad_suff, beta_n 

    def Inner_iter(self, mapping, N_fom, Adjoint):
        v, gJ= self.Array[0], self.Array[1]  
        beta, alpha= self.Array[4], self.Parameters[0]
        v_new, X = self.Design_update(v, alpha, gJ, mapping, beta, is_init=True)
        # gJ= npa.where(((X>=0)&(gJ<0)),0, npa.where(((X>=1)&(gJ>0)),0,gJ))
        if self.Array[4]== npa.inf:
            #--------| Forward simulation |---------#
            f0, f0s = Adjoint(X, N_fom, Case=False) # 
            #---------------------------------------#
            self.Parameters[1]=f0 
            self.numevl +=1
            self.evaluation_history.append(self.Parameters[1])
            print("\n First FoM: {}".format(self.evaluation_history[1]))      
            print("Last FoM: {}".format(self.Parameters[1]))                        
            np.savetxt(self.design_dir+"lastdesign.txt", npa.clip(X, 0.0, 1.0))    
            np.savetxt(self.design_dir+"last_v.txt", v_new)                    
            print("----------Optimization Complete-----------")
            print(f"Full BT cnt: {self.bt_max_cnt} \n")   
        else:       
            if self.is_converged: 
                f_old= self.Parameters[2]  
            else:                         
                f_old= self.Parameters[1] 
            v_new, X, gJ, grad_suff, beta = self.Conversion(mapping, v_new, X)
            #-------| Forward simulation |----------#
            f0, f0s = Adjoint(X, N_fom, Case=False) #
            #---------------------------------------#
            if self.Raw:
                Armijo_cond=-1
            else: 
                Armijo_cond= f_old+ (1/X.size)*(1/X.size)*alpha*grad_suff
            if (f0 < Armijo_cond or _is_failed_fom(f0)) and self.cur_iter[0]>1:     
                alpha_scale, bt_cnt= 0.1, 0
                if grad_suff==0:
                    alpha_scale=0.01        
                while bt_cnt < self.bt_tol and (f0 < Armijo_cond or _is_failed_fom(f0)):        
                    if bt_cnt == 0:       
                        foms_info,fom_info,bt_h,trig,stop= Back_tracking_init(f0, f0s, self.bt_tol) 
                    bt_cnt +=1            
                    alpha *=alpha_scale   
                    v_new, X = self.Design_update(v, alpha, gJ, mapping, beta)
                    #---------| Forward simulation in Backtracking processes |----------#
                    fom_info[bt_cnt], foms_info[bt_cnt] = Adjoint(X, N_fom, Case=False) #
                    #-------------------------------------------------------------------#
                    trig, stop= Back_tracking(fom_info, bt_cnt, trig, self.is_converged) 
                    bt_h= Back_traking_record(bt_h, fom_info, bt_cnt)             
                    if stop and trig:     
                        alpha *= (1/alpha_scale)**bt_cnt                          
                        v_new, X = self.Design_update(v, alpha, gJ, mapping, beta)
                        f0, f0s=Back_tracking_call(fom_info, foms_info, 0)        
                        if alpha_scale <1:                   
                            alpha_scale= (1/alpha_scale)                          
                            bt_hd, bt_cnt= bt_h, 0            
                            print("Restart backtraking with upscaling factor")    
                        else:             
                            print(f"Continue without rescaling: {alpha}")        
                            self.Parameters[3]=2
                            break         
                    elif stop:            
                        alpha *= (1/alpha_scale)                                  
                        v_new, X = self.Design_update(v, alpha, gJ, mapping, beta)
                        f0, f0s=Back_tracking_call(fom_info, foms_info, bt_cnt-1) 
                        print(f"Continue with {bt_cnt-1}steps reduced LR: {alpha}")
                        break         
                    else:                 
                        f0, f0s = Back_tracking_call(fom_info, foms_info, bt_cnt) 
                        print(f"Backtraking: {bt_cnt}steps reduced LR: {alpha}")
                        if bt_cnt==self.bt_tol-1:
                            break
                    #-----------------------------------------------------------------#
                if _is_failed_fom(f0):
                    print(
                        "[optimizer] all reduced steps remained unstable; "
                        "rejecting this update and keeping the previous geometry."
                    )
                    self.Parameters[0] = alpha
                    self.Parameters[1] = f_old
                    self.Array[0] = v
                    self.Array[1] = gJ * 0
                    self.Array[4] = beta
                    return
                if self.Re_roll and self.Outer_M[1] <6 and f0 < f_old*0.9:
                    pass
                else:
                    if bt_cnt==self.bt_tol-1:
                        print(f"Full {bt_cnt}steps reduced LR: {alpha}")
                        #-------| Adjoint simulation with final alpha |---------#
                        dJ_dus = Adjoint("Adj", N_fom, Case=True)               #
                        dJ_dus= Born_validity(dJ_dus, N_fom, top_k=self.Born_k) #
                        #-------------------------------------------------------#
                        self.bt_max_cnt +=1
                    else:
                        #----| Forward & Adjoint simulation with best alpha |----#
                        f0, f0s, dJ_dus = Adjoint(X, N_fom, Case=True)           #
                        dJ_dus= Born_validity(dJ_dus, N_fom, top_k=self.Born_k)  #
                        #--------------------------------------------------------#                     
                if alpha_scale < 1:
                    bt_hd=bt_h   
                    bt_h= [f0]*self.bt_tol
                for i in range(self.bt_tol):        
                    if self.Re_roll:
                        self.wrong_evaluation_history[i][-1]=bt_hd[i]
                        self.wrong_evaluation_history2[i][-1]=bt_h[i]
                    else:
                        self.wrong_evaluation_history[i].append(bt_hd[i]) 
                        self.wrong_evaluation_history2[i].append(bt_h[i])  
            else:
                if self.Re_roll and self.Outer_M[1] <6 and f0 < f_old*0.9:
                    pass
                else:
                    #---------------| Adjoint simulation |------------------#
                    dJ_dus = Adjoint("Adj", N_fom, Case=True)               #
                    dJ_dus= Born_validity(dJ_dus, N_fom, top_k=self.Born_k) #
                    #-------------------------------------------------------#                    
                for i in range(self.bt_tol):        
                    if self.Re_roll:
                        self.wrong_evaluation_history[i][-1]=(f0)  
                        self.wrong_evaluation_history2[i][-1]=(f0)        
                    else:
                        self.wrong_evaluation_history[i].append(f0)                   
                        self.wrong_evaluation_history2[i].append(f0)
            self.Parameters[0]= alpha     
            self.Parameters[1]= f0        
            self.Array[0]= v_new  
            self.Array[4]= beta
            if self.Re_roll and self.Outer_M[1] <6 and f0 < f_old*0.9:
                return              
            #---------------------------------------------------------------------------------------------#
            dJ_du = Adjoint([dJ_dus, self.Array[2]], f0s, Case=3)           #| Gradient analysis (Case 3) |
            # if self.Raw:
            # g_m=dJ_du
            # else:
            # Momentum (Adam) ----------------------------------------------------------------------------|
            g_m, self.Array[2], self.Array[3]= Momentum(dJ_du, self.Array[2], self.Array[3], self.numevl) #
            # Jacobian grad ------------------------------------------------------------------------------|
            if v.size > 0:                
                gradient = v*0            
                gradient[:] = tensor_jacobian_product(mapping, 0)(v_new, beta, g_m)# backprop #
                gradient = npa.where(npa.isfinite(gradient), gradient, 0)
                denorm_avg=npa.mean(npa.abs(gradient))
                denorm_max=npa.max(npa.abs(gradient))
                denorm_avg=npa.where(npa.isfinite(denorm_avg), denorm_avg, 0)
                denorm_max=npa.where(npa.isfinite(denorm_max), denorm_max, 0)
                self.Outer_M[2] = npa.sum(npa.where(((1e-3<X)&(X<(1-1e-3))), 0, 1))/X.size

            if self.Re_roll:
                self.grad_mean_history[-1]=denorm_avg          
                self.grad_max_history[-1]=denorm_max             
                self.learning_rate_history[-1]=(alpha)  
                self.evaluation_history[-1]=(f0) 
                self.binarization_history[-1]=self.Outer_M[2]
            else:
                self.grad_mean_history.append(denorm_avg)            
                self.grad_max_history.append(denorm_max)              
                self.learning_rate_history.append(alpha)  
                self.evaluation_history.append(f0) 
                self.binarization_history.append(self.Outer_M[2])  
                self.numevl += 1              
                self.cur_iter[0] += 1      
            print("\nFirst FoM: {}".format(self.evaluation_history[1]))
            print("Best FoM: {}".format(self.Best[6]))
            print("Local Best FoM: {}".format(self.Best[0]))
            print("Current FoM: {}".format(f0))
            print("Current beta: ", beta)       
            print(f"Max dv: {denorm_max}, Mean dv: {denorm_avg}")
            print(f"Binarization rate: {round(self.Outer_M[2]*100,2)}%") 
            print(f"Learning rate: {(alpha)}\n")                                     
            if denorm_max <= 0:
                print("[optimizer] zero/non-finite gradient; using zero update direction")
                self.Array[1]= gradient*0
            else:
                self.Array[1]= 0.5*gradient/denorm_max    

    """ Total loop """
    def __call__(self, mapping, N_fom, Adjoint):
        inner_iter = 0
        max_outer_iters = int(os.environ.get("MSOPT_MAX_OUTER_ITERS", "777"))
        self.Is_gray= mapping.Is_freeform[1]
        while inner_iter < max_outer_iters:
            self.Updater()
            self.Inner_iter(mapping, N_fom, Adjoint)
            if self.Array[4] == npa.inf:
                break
            if self.is_converged:
                self.After_Conv()
                if not self.Re_roll:
                    self.Warm_restarter() 
            else:
                self.Warm_restarter()
            if self.Parameters[2] !=-777:
                if not self.Re_roll:
                    self.Conv_check()   
            inner_iter +=1
        np.savetxt(self.design_dir+"evaluation.txt", self.evaluation_history)
        np.savetxt(self.design_dir+"evaluation2.txt", self.wrong_evaluation_history)
        np.savetxt(self.design_dir+"evaluation3.txt", self.wrong_evaluation_history2)
        np.savetxt(self.design_dir+"learning_rate.txt", self.learning_rate_history)
        np.savetxt(self.design_dir+"binarization.txt", self.binarization_history)
        np.savetxt(self.design_dir+"grad_mean.txt", self.grad_mean_history)
        np.savetxt(self.design_dir+"grad_max.txt", self.grad_max_history)
        np.savetxt(self.design_dir+"beta.txt", self.beta_history)
         



"""
Calculation methods for FoM
C1. Cross_product: Cross product for power calculation <---- need to fix 
C2. Substract_field: Substract the fields to calculate the scattered fields <- needs pre-simulation for total fields
C3. Overlap_intg: Overlap integration to calculate purity of current field profile <- needs pre-simulation for target fields
"""
""" C1 """
def Cross_product(E, H, axis=2):
    S=[0]*3
    S[0]= npa.real(E[1]*npa.conjugate(H[2]) - E[2]*npa.conjugate(H[1]))
    S[1]= npa.real(E[2]*npa.conjugate(H[0]) - E[0]*npa.conjugate(H[2]))
    S[2]= npa.real(E[0]*npa.conjugate(H[1]) - E[1]*npa.conjugate(H[0]))
    return npa.abs(npa.sum(S[axis]))

""" C2 """
def Substract_field(Total_field, Scattered_field):
    Substracted_field= [0]*3
    Substracted_field[0]= Scattered_field[0]-Total_field[0]
    Substracted_field[1]= Scattered_field[1]-Total_field[1]
    Substracted_field[2]= Scattered_field[2]-Total_field[2]
    return Substracted_field

""" C3 """
def Overlap_intg(Target, Output, normalization=False, Reflection=False, Phase_lock=False, self_norm=0):
    if Reflection:
        X=Output[0]*(Target[0])
        Y=Output[1]*(Target[1])
        Z=Output[2]*(Target[2])
    else:
        X=Output[0]*npa.conjugate(Target[0])
        Y=Output[1]*npa.conjugate(Target[1])
        Z=Output[2]*npa.conjugate(Target[2])
    FoM=(npa.sum(X+Y+Z))    
    if normalization:
        Tn=(npa.abs(Target[0])**2) + (npa.abs(Target[1])**2) + (npa.abs(Target[2])**2)
        if self_norm==0:
            On= (npa.abs(Output[0])**2) + (npa.abs(Output[1])**2) + (npa.abs(Output[2])**2)
        else:
            On= self_norm
        Purity=npa.abs(FoM)**2/((npa.sum(Tn))*(npa.sum(On)))
        if Phase_lock:
            phasor=FoM/(npa.abs(FoM)+1e-30)
            phase_term= npa.real(phasor)
            return Purity*0.8+phase_term*0.2
        else:
            return Purity
    else:
        return npa.abs(FoM)**2

""" C4 """
def Overlap_intg_cyl(Target, Output, dr, Reflection=False, Is_Radial=False):
    Overlap, T_nom, O_nom = 0, 0, 0
    if not Reflection:
        Target=npa.conjugate(Target)
    for i in [0, 1, 2]:
        if Is_Radial:
            Overlap +=npa.sum(Output[i]*(Target[i])*dr)
            T_nom +=npa.sum(npa.abs(Target[i])**2)*dr
            O_nom +=npa.sum(npa.abs(Output[i])**2)*dr
        else:
            Overlap +=npa.dot(Output[i]*(Target[i]),dr)
            T_nom +=npa.dot(npa.abs(Target[i])**2, dr)
            O_nom +=npa.dot(npa.abs(Output[i])**2, dr)
    denom=npa.abs(Overlap)**2
    nom = T_nom*O_nom + 1e-8
    return denom/nom

""" C5 """
def Flux_cyl(E, H, dr, axis=2):
    S=[0]*3
    S[0]= npa.real(E[1]*npa.conjugate(H[2]) - E[2]*npa.conjugate(H[1]))
    S[1]= npa.real(E[2]*npa.conjugate(H[0]) - E[0]*npa.conjugate(H[2]))
    S[2]= npa.real(E[0]*npa.conjugate(H[1]) - E[1]*npa.conjugate(H[0]))
    if axis==2:
        return npa.dot(S[axis],dr)
    elif axis==1:
        return dr*npa.sum(S[axis])/(0.5*np.pi)
    else:
        return npa.sum(S[axis]*dr)

""" C6 """
def Overlap_intg_comp(Target, Output, Phase_target, Reflection=False):
    if Reflection:
        X=Output[0]*(Target[0])
        Y=Output[1]*(Target[1])
        Z=Output[2]*(Target[2])
    else:
        X=Output[0]*npa.conjugate(Target[0])
        Y=Output[1]*npa.conjugate(Target[1])
        Z=Output[2]*npa.conjugate(Target[2])
    FoM=npa.sum(X+Y+Z)
    Phase= npa.angle(FoM)
    Magnitude= npa.abs(FoM)
    FoM_phase= Magnitude**2 * npa.cos(Phase - Phase_target)**2
    Tn=(npa.abs(Target[0])**2) + (npa.abs(Target[1])**2) + (npa.abs(Target[2])**2)
    On= (npa.abs(Output[0])**2) + (npa.abs(Output[1])**2) + (npa.abs(Output[2])**2)
    return FoM_phase/((npa.sum(Tn))*(npa.sum(On)))



""" 
Mapping class: more details in Sub_Mapping
"""
class Mapping:
    def __init__(
        self,
        Symmetry_sim = False,
        Sym_geo_width =False,
        Sym_geo_length=False,
        Sym_geo_C2 = False,
        Sym_geo_C8 = False,
        Sym_offdiag= False,
        Is_waveguide=[False, False, False, 2], # Is_wg?, Is_middle?, Is_3D?, number of local region
        DR_info = [None, None, None, 1, 2, 0], # Dx:0, Dy:1, Dz:2, width-idx, length-idx, hight-idx
        DR_N_info=[None, None, None, None], # Nx, Ny, Nz,  resolution
        Is_Cylindrical=[False, False, 0.0], # Is_cyl?, Is_2D?, Disk_R
        Mask_info=[None, None], #  wg_left mask sz, wg_right mask sz
        Sub_pixels=[False, 0],
        Mask_pixels=0,
        MFS= None,
        MGS= None,
        Is_1D_grating=[False, False], # Is 1D grating?, Is slanted?
        Is_freeform=[False, False, False], # Is Free-form?, Is gray-scale?, Is Single layer?
        Is_slanted_grating=False,
    ):
        self.MFS = MFS
        self.MFS0 =MFS
        self.MGS = MGS
        self.MGS0 =MGS
        self.DR_res =DR_N_info[3]
        self.Mask_info = Mask_info
        self.Sub_pixels =Sub_pixels
        self.Mask_pixels=Mask_pixels
        self.Sym_geo_C2 = Sym_geo_C2
        self.Sym_geo_C8 = Sym_geo_C8
        self.Sym_offdiag = Sym_offdiag
        self.Is_freeform = Is_freeform
        self.Is_waveguide= Is_waveguide
        self.Symmetry_sim = Symmetry_sim
        self.Sym_geo_width =Sym_geo_width
        self.Sym_geo_length=Sym_geo_length
        self.Is_1D_grating = Is_1D_grating
        self.Is_Cylindrical= Is_Cylindrical
        self.slanted= Is_slanted_grating
        self.N_height=DR_N_info[DR_info[5]]
        if Is_freeform[0] and not Is_freeform[2]:
            print("\n This is Freeform topology optimization \n")
            return
        if DR_info[5] == DR_info[4] or DR_info[5] == DR_info[3]:
            raise ValueError("Design layer includes height axis")
        elif DR_info[3] >= DR_info[4] and DR_info[3] !=2:
            raise ValueError("Design layer width axis should be low axis")
        else: # width axis < length axis && layer size = width*length
            if Is_waveguide[0]:            
                if Is_waveguide[1]:
                    self.MGS0= MGS*0.5
                    self.MFS0=self.MFS + MGS*0.5
                    self.Mask_info[0] += MGS*0.5 
                    self.Mask_info[1] += MGS*0.5
                    if Is_waveguide[3]/2==int(Is_waveguide[3]/2):
                        print("Reference layer: middle layer")
                    else: # total layer number should be even (mid)
                        self.Is_waveguide[3] = Is_waveguide[3]+1
                        print("Reference layer: middle+1 layer")
                self.DR_width = DR_info[DR_info[3]]
                self.N_width= DR_N_info[DR_info[3]]
                self.DR_length= DR_info[DR_info[4]]
                self.N_length=DR_N_info[DR_info[4]]
            else: # Cylindrical grating use the width axis only
                if not Is_Cylindrical[0]:
                    self.DR_length =DR_info[DR_info[4]]
                    self.N_length=DR_N_info[DR_info[4]]
                self.DR_width = DR_info[DR_info[3]]
                self.N_width =DR_N_info[DR_info[3]]
                self.Mask_info[0]=0
                self.Mask_info[1]=0
    
    def __call__(
        self,
        x,
        beta,
        Is_opt=True,
    ) -> np.ndarray:
        if Is_opt:  # Filter & Projection based Mapping function
            if self.Is_freeform[0]:
                if beta>2:
                    eee=-0.5
                else:
                    eee=0.5
                if self.Is_freeform[1]:
                    if self.Is_freeform[2]:
                        if self.N_height >1:
                            x_copy= Sub_Mapping.Vertical_sidewall(
                                Reference_layer = x,
                                N_height = self.N_height,
                            )
                            x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
                            return x_copy.flatten()
                    else:
                        return x
                elif self.Is_freeform[2]:
                    if self.Sym_geo_C8:
                        x_ref = npa.reshape(x,(self.N_width,self.N_length))
                        x_ref = (npa.fliplr(x_ref) + x_ref)/2
                        x_ref = (npa.flipud(x_ref) + x_ref)/2
                        x_ref = (x_ref.transpose() + x_ref)/2
                        x = x_ref.flatten() 
                    x=Sub_Mapping.tanh_projection_m(x, beta, eee)
                    if self.N_height >1:
                        x_copy= Sub_Mapping.Vertical_sidewall(
                            Reference_layer = x,
                            N_height = self.N_height,
                        )
                        x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
                    return Sub_Mapping.tanh_projection_m(x_copy, beta, 0.5).flatten()
                else:
                    xx=Sub_Mapping.tanh_projection_m(x, beta, eee)
                    return Sub_Mapping.tanh_projection_m(xx, beta, 0.5).flatten()
            else:
                if self.Is_Cylindrical[0]: # 1D Grating structure
                    x_copy= Sub_Mapping.Grating(
                        Disk_R= self.Is_Cylindrical[2],
                        Is_Cyl= self.Is_Cylindrical[1],
                        Mask_pixels=self.Mask_pixels,
                        Min_size_top = self.MFS0,
                        DR_width= self.DR_width,
                        N_width= self.N_width,
                        DR_res= self.DR_res,
                        Min_gap=self.MGS0,
                        beta= beta,
                        x = x,
                    )
                elif self.Is_1D_grating[0]:
                    peudo_width= round(2*(self.MGS + self.MFS),1)
                    peudo_N_width= int(self.DR_res * peudo_width)+1
                    x_copy= Sub_Mapping.get_reference_layer_1D(
                        is_slanted=self.Is_1D_grating[1],
                        Min_size_top = self.MFS,
                        DR_length= self.DR_length,
                        N_length= self.N_length,
                        peudo_width= peudo_width,
                        peudo_N_width= peudo_N_width,
                        DR_res= self.DR_res,
                        Min_gap=self.MGS,
                        beta= beta,
                        x = x,
                    )
                    if self.Is_1D_grating[1]:
                        x_copy= Sub_Mapping.Slant_sidewall(
                            Number_of_local_region=self.Is_waveguide[3],
                            is_Middle = self.Is_waveguide[1],
                            Reference_layer = x_copy,
                            N_height = self.N_height,
                            DR_width = peudo_width,
                            DR_length=self.DR_length,
                            DR_res = self.DR_res,
                            Min_gap= self.MGS,
                            beta = beta,
                        )
                        x_copy = npa.reshape(x_copy,(self.N_height,peudo_N_width,self.N_length))
                        x_copy = x_copy[:,int(peudo_N_width/2),:].flatten()
                        x_copy = npa.reshape(x_copy,(self.N_height,self.N_length)).transpose()
                else:   # 2D Grating structure / 3D structure
                    x_copy = Sub_Mapping.get_reference_layer(
                        Symmetry_in_Sim = self.Symmetry_sim,
                        Width_symmetry = self.Sym_geo_width,
                        Length_symmetry=self.Sym_geo_length,
                        C2_symmetry = self.Sym_geo_C2,
                        QR_symmetry= self.Sym_offdiag,
                        Pseudo_Cyl= self.Sym_geo_C8,
                        DR_length = self.DR_length,
                        DR_width = self.DR_width,
                        N_length= self.N_length,
                        N_width =self.N_width,
                        DR_res =self.DR_res,
                        Is_waveguide=self.Is_waveguide[0],
                        Mask_pixels= self.Mask_pixels,
                        input_w_top=self.Mask_info[0],
                        Sub_pixels= self.Sub_pixels,
                        w_top = self.Mask_info[1],
                        Min_size_top=self.MFS0,
                        Min_gap=self.MGS0,
                        beta= beta,
                        x = x,
                    )
                    if self.Is_waveguide[2]:# Slanted Waveguide
                        x_copy= Sub_Mapping.Slant_sidewall(
                            Number_of_local_region=self.Is_waveguide[3],
                            is_Middle = self.Is_waveguide[1],
                            Reference_layer = x_copy,
                            N_height = self.N_height,
                            DR_width = self.DR_width,
                            DR_length=self.DR_length,
                            DR_res = self.DR_res,
                            Min_gap= self.MGS,
                            beta = beta,
                        )
                    elif self.N_height >1:
                        if self.slanted:
                            x_copy= Sub_Mapping.Slant_sidewall(
                                Number_of_local_region=2,
                                is_Middle = False,
                                Reference_layer = x_copy,
                                N_height = self.N_height,
                                DR_width = self.DR_width,
                                DR_length=self.DR_length,
                                DR_res = self.DR_res,
                                Min_gap= self.MGS,
                                beta = beta,
                            )
                        else:
                            x_copy= Sub_Mapping.Vertical_sidewall(
                                Reference_layer = x_copy,
                                N_height = self.N_height,
                            )
                        x_copy = npa.reshape(x_copy,(self.N_height,self.N_width*self.N_length)).transpose()
        else: # 2D -> 3D projection of optimized reference layer
            if self.Is_waveguide[2]: # Is Slanted 3D Waveguide?
                x_copy= Sub_Mapping.Slant_sidewall(
                    Number_of_local_region=self.Is_waveguide[3],
                    is_Middle= self.Is_waveguide[1],
                    Reference_layer = x,
                    N_height = self.N_height,
                    DR_width = self.DR_width,
                    DR_length=self.DR_length,
                    DR_res= self.DR_res,
                    Min_gap= self.MGS,
                    beta = beta,
                )
        return x_copy.flatten()
