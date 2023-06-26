import math
import numpy as np

def F_g(eng1:float,eng2:float):
    if eng2!=None:
        return 2.83e-11/1.62*(pow(eng1,-1.62)-pow(eng2,-1.62))
    else:
        return 2.83e-11/1.62*pow(eng1,-1.62)
    
def F_g1(eng1:float,eng2:float):
    if eng2!=None:
        return 1.1e-11/(pow(0.15,-3.2)*2.2)*(pow(eng1,-2.2)-pow(eng2,-2.2))
    else:
        return 1.1e-11/(pow(0.15,-3.2)*2.2)*pow(eng1,-2.2)
    
def F_g2(eng1:float,eng2:float):
    if eng2!=None:
        return 2e-11/(pow(0.15,-2.9)*1.9)*(pow(eng1,-1.9)-pow(eng2,-1.9))
    else:
        return 2e-11/(pow(0.15,-2.9)*1.9)*pow(eng1,-1.9)
    
def F_p(eng1:float,eng2:float):
    # return 0.898e-5*pow((eng1+eng2)/2,-2.7)

    # if eng2!=None:
    #     total=0
    #     for i in np.arange(eng1,eng2,0.01):
    #         # total=total+0.898*1e-15*math.pow(10,-(math.log10(i)+1)/0.7)*0.01
    #         total=total+0.898e-5*pow(i,-2.7)*0.01
    #     return total
    # else:
    #     total=0
    #     for i in np.arange(eng1,1000,1):
    #         # total=total+0.898*1e-15*math.pow(10,-(math.log10(i)+1)/0.7)*1
    #         total=total+0.898e-5*pow(i,-2.7)*1
    #     return total



    if eng2!=None:
        return 0.898e-5/1.7*(pow(eng1,-1.7)-pow(eng2,-1.7))
    else:
        return 0.898e-5/1.7*pow(eng1,-1.7)

class Info(object):
    def __init__(self):
        self.Eng=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,4.,5.,6.,7.,8.,9.]
        # self.Eff_g=[0.045,0.088,0.141,0.171,0.202,0.234,0.262,0.291,0.307,0.317,0.422,0.529,0.579,0.633,0.681,0.699,0.716,0.734]
        # self.Eff_p=[0.037,0.082,0.137,0.171,0.208,0.243,0.274,0.307,0.326,0.337,0.453,0.571,0.616,0.665,0.710,0.735,0.761,0.790]
        self.Eff_g=[0.347,0.695,0.838,0.891,0.912,0.921,0.928,0.927,0.930,0.932,0.934,0.935,0.936,0.937,0.937,0.938,0.938,0.939]
        self.Eff_p=[0.029,0.129,0.268,0.408,0.519,0.621,0.697,0.752,0.796,0.824,0.915,0.930,0.934,0.936,0.937,0.938,0.938,0.939]
        # self.Q=[1.10,1.30,1.50,1.75,2.00,2.20,2.30,2.40,2.55,2.70,3.2,3.5,3.6,3.7,3.8,3.9,4.0,4.05]
        self.Q=[1.05,1.36,2.22,2.64,3.25,3.81,4.01,4.34,4.54,5.00,5.41,5.58,5.60,5.61,5.62,5.60,5.20,4.87]
        self.Angel=[0.39563173084001013,0.3360615488873496,0.29184233581693086,0.27268062165758183,0.26587142406744446,0.25597426683431485,0.24703818414687853,0.24108325125765625,0.243033893512969,0.24256633395396996,0.2402738107716347, 0.2309665798324964, 0.2290586303459129, 0.22788317654460833, 0.22205294706545175, 0.22496408873724595, 0.22027175830686505, 0.2165419186013675]
        # self.Angel=[0.8651279264223918,0.7227780746230985,0.5955153195922451,0.5846971532520784,0.5359434208767576,0.5230381991874321,0.5079652284338684,0.48968165587054246,0.4964155431966568,0.5193946222594595,0.4402738107716347, 0.4309665798324964, 0.3290586303459129, 0.35788317654460833, 0.34205294706545175, 0.34496408873724595, 0.34027175830686505, 0.3465419186013675, 0.34540296979419523]
        
        # self.CosAngel=[0.99987889,0.9999233,0.9999503,0.99996345,0.999972060,0.9999775,0.99998048,0.99998558,0.99998837,0.99999039]
        # self.CosAngel=[0.999984,0.999987,0.999990,0.999990,0.999972,0.999978,0.999980,0.999986,0.999988,0.999990]
        # self.CosAngel=[math.cos(self.Angel[i]*math.pi/180) for i in range(len(self.Angel))]
        self.CosAngel=[math.cos(self.Angel[i]*math.pi/180) for i in range(len(self.Angel))]
        self.index=0
    
    def get_next_energy(self,eng:float):
        if eng<self.Eng[0]:
            self.index=-1
            return self.Eng[0]
        for i in range(len(self.Eng)-1):
            if self.Eng[i]<=eng<self.Eng[i+1]:
                self.index=i
                return self.Eng[i+1]
            # if i==(len(self.Eng)-2):
            #     print(i,len(self.Eng)-2)
            #     break
            # else:
            #     print(i,len(self.Eng)-2)
        if eng>=self.Eng[-1]:
            self.index=len(self.Eng)-1
            return None
        
            
    def get_eff_g(self):
        if self.index==-1:
            return self.Eff_g[0]
        if self.index==len(self.Eff_g)-1:
            return self.Eff_g[self.index]
        else:
            return (self.Eff_g[self.index]+self.Eff_p[self.index+1])/2
        
    def get_eff_p(self):
        if self.index==-1:
            return self.Eff_p[0]
        if self.index==len(self.Eff_p)-1:
            return self.Eff_p[self.index]
        else:
            return (self.Eff_p[self.index]+self.Eff_p[self.index+1])/2
    
    def get_q(self):
        if self.index==0:
            return self.Q[0]
        if self.index==len(self.Q)-1:
            return self.Q[self.index]
        else:
            return (self.Q[self.index]+self.Q[self.index+1])/2
    
    def get_cos(self):
        if self.index==0:
            return self.CosAngel[0]
        if self.index==len(self.CosAngel)-1:
            return self.CosAngel[self.index]
        else:
            return (self.CosAngel[self.index]+self.CosAngel[self.index+1])/2


def N_g(eng1:float,eng2:float):
    N_total=0
    info=Info()
    temp_eng1=eng1
    temp_eng2=0
    while True:
        temp_eng2=info.get_next_energy(temp_eng1)
        eff=info.get_eff_g()
        q=info.get_q()
        N_total=N_total+F_g(temp_eng1,temp_eng2)*eff*800*800*1e4*q
        # N_total=N_total+F_g(temp_eng1,temp_eng2)

        if temp_eng2==eng2 or temp_eng2==None:
            break
        else:
            temp_eng1=temp_eng2
    return N_total

def N_g1(eng1:float,eng2:float):
    N_total=0
    info=Info()
    temp_eng1=eng1
    temp_eng2=0
    while True:
        temp_eng2=info.get_next_energy(temp_eng1)
        eff=info.get_eff_g()
        N_total=N_total+F_g1(temp_eng1,temp_eng2)*eff*800*800*1e4

        if temp_eng2==eng2 or temp_eng2==None:
            break
        else:
            temp_eng1=temp_eng2
    return N_total

def N_g2(eng1:float,eng2:float):
    N_total=0
    info=Info()
    temp_eng1=eng1
    temp_eng2=0
    while True:
        temp_eng2=info.get_next_energy(temp_eng1)
        eff=info.get_eff_g()
        N_total=N_total+F_g2(temp_eng1,temp_eng2)*eff*800*800*1e4

        if temp_eng2==eng2 or temp_eng2==None:
            break
        else:
            temp_eng1=temp_eng2
    return N_total

def N_p(eng1:float,eng2:float):
    N_total=0
    info=Info()
    temp_eng1=eng1
    temp_eng2=0
    while True:
        temp_eng2=info.get_next_energy(temp_eng1)
        eff=info.get_eff_p()
        cos_angel=info.get_cos()
        N_total=N_total+F_p(3*temp_eng1,3*temp_eng2 if temp_eng2!=None else None)*eff*800*800*1e4*2*math.pi*(1-cos_angel)

        if temp_eng2==eng2 or temp_eng2==None:
            break
        else:
            temp_eng1=temp_eng2
    return N_total


def calc_crab_pulsar(eng1:float,eng2:float=None):
    # eng1=0.95
    # eng2=None
    g=N_g(eng1,eng2)
    g1=N_g1(eng1,eng2)
    g2=N_g2(eng1,eng2)
    p=N_p(eng1,eng2)

    Nbac=g+p
    year=320*3600
    NP1sum=g1
    NP2sum=g2

    Non1 = ( Nbac * year * 0.043  + NP1sum * year )
    Non2 = ( Nbac * year * 0.045 + NP2sum * year )
    Noff =   Nbac * year * 0.35

    alp1=0.043/0.35
    alp2=0.045/0.35

    sigma1=(Non1-alp1*Noff)/math.sqrt(alp1*(Non1+Noff))
    sigma2=(Non2-alp2*Noff)/math.sqrt(alp2*(Non2+Noff))

    print(sigma1,sigma2)
    print(Non1-alp1*Noff,Non2-alp2*Noff)

    return sigma1,sigma2,Non1-alp1*Noff,Non2-alp2*Noff
