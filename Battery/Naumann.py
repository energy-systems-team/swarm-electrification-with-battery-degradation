"""
Naumann Model functions

Sources:

M. Naumann, M. Schimpe, P. Keil, H. C. Hesse, A. Jossen, 
Analysis and modeling of calendar aging of a commercial lifepo4/graphite cell, 
Journal of Energy Storage 17 (2018) 153 169. 
doi:https://doi.org/10.1016/j.est.2018.01.019. 
URL https://www.sciencedirect.com/science/article/pii/S2352152X18300665

M. Naumann, F. B. Spingler, A. Jossen,
Analysis and modeling of cycle aging of a commercial lifepo4/graphite cell,
Journal of Power Sources 451 (2020) 227666.
doi:https://doi.org/10.1016/j.jpowsour.2019.227666.
URL https://www.sciencedirect.com/science/article/pii/S037877531931659
"""

import numpy as np


def init_naumann(self):
    self.a_cyc = 0.063
    self.b_cyc = 0.0971
    self.c_cyc = 4.0253
    self.d_cyc = 1.0923
    self.z_cyc = 0.5
    self.FEC = 1e-20
    self.kref = 0.0012571
    self.Ea_cal = 17126.0
    self.Tref = 298.15
    self.c_cal = 2.8575
    self.d_cal = 0.60225
    self.Q_cal_loss = 0.0
    self.Q_cyc_loss = 0.0

def update_naumann(self, charge_power, discharge_power, include_float_degradation=True, dt_hours=1.0, temperature=298.15):
    dt_seconds = dt_hours * 3600.0

    # === CYCLIC aging (shared detection)
    flip = self.cycle_check(charge_power, discharge_power, dt_hours)
    if flip["completed"]:
        DOC = flip["DOC"]
        hours_active = flip["active_hours"]
        C_rate = DOC / max(hours_active, 1e-9)

        # Project current damage into a "virtual" FEC* to handle changing stress factors
        k_rate = self.a_cyc * C_rate + self.b_cyc
        k_DOC  = self.c_cyc * (DOC - 0.6)**3 + self.d_cyc
        k_cyc_total = max(k_rate * k_DOC, 1e-30)
        
        FEC_star = (self.Q_cyc_loss / k_cyc_total)**(1.0/self.z_cyc)
    
        dFEC = 0.5 * DOC                                 # half-cycle increment

        # Integrated form for the increment to avoid singularity at FEC=0
        dQ_cyc = k_cyc_total * ((FEC_star + dFEC)**self.z_cyc - (FEC_star**self.z_cyc))
    
        self.Q_cyc_loss += dQ_cyc
        self.FEC += dFEC
    else:
        dQ_cyc = 0.0

    # === CALENDAR aging (every time step) ===
    if include_float_degradation:
        # Since T=Tref, the Arrhenius term exp(...) is 1. k_temp is just k_ref.
          
        k_temp = self.kref * np.exp(-self.Ea_cal/self.R * (1.0/temperature - 1.0/self.Tref))
        
        SOC_safe = self.ch / (self.C + 1e-10)
        SOC_safe = min(1.0, max(0.0, SOC_safe))
        k_SOC = self.c_cal * (SOC_safe - 0.5)**3 + self.d_cal
                   
        # Use virtual time t* to handle changing stress factors
        k_cal_total = max(k_temp * k_SOC, 1e-30)
        t_star = (self.Q_cal_loss / k_cal_total)**2

        # Integrated form for the increment to correctly model t^0.5 dependence
        dQ_cal = k_cal_total * (np.sqrt(t_star + dt_seconds) - np.sqrt(t_star))
        self.Q_cal_loss += dQ_cal
    else:
        dQ_cal = 0.0
    
    aging_factor_cyc = dQ_cyc /100
    aging_factor_float = dQ_cal /100
    
    aging_factor = aging_factor_cyc + aging_factor_float
    self.aging_factor_history = np.append(self.aging_factor_history, aging_factor)
    self.aging_factor_cyclic_history = np.append(self.aging_factor_cyclic_history, aging_factor_cyc)
    self.aging_factor_float_history = np.append(self.aging_factor_float_history, aging_factor_float)

    if self.include_degradation:
        total_q_loss_percent = self.Q_cal_loss + self.Q_cyc_loss
        self.C = max(0.0, self.C_init * (1.0 - total_q_loss_percent / 100.0))
        MAX_C = self.C
        self.ch = min(self.ch, MAX_C)
        
    delta_SOH = (self.capacity_history[-1] - self.C)/(max(self.C_init,1e-30)*(1-self.END_OF_LIFE))
    self.deltaSOH_history = np.append(self.deltaSOH_history, delta_SOH)

    # Attribute the step's ΔSOH by contribution
    total_step = aging_factor_float + aging_factor_cyc
    if total_step <= 0:
        self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, 0.0)
        self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, 0.0)
    else:
        # Portion of the SOH drop due to calendar vs cycle this step
        delta_SOH_float = delta_SOH * (aging_factor_float / total_step)
        delta_SOH_cyc   = delta_SOH * (aging_factor_cyc   / total_step)
        self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, delta_SOH_float)
        self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, delta_SOH_cyc)
   
