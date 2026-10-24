"""
Gasper Model functions

Source:

P. Gasper, N. Collath, H. C. Hesse, A. Jossen, K. Smith,
Machine learning assisted identification of accurate battery lifetime models with uncertainty,
Journal of The Electrochemical Society 169 (8) (2022) 080518.
doi:10.1149/1945-7111/ac86a8.
URL https://doi.org/10.1149/1945-7111/ac86a8
"""        
import numpy as np
from math import erf


def init_gasper(self):
    # Calendar
    self.q1_a = 9.90e-01
    self.q1_b = -2.88e+06
    self.q1_c = 8.74e+03
    self.q2   = 1.31e-04
    self.q3_a = 3.33e-04
    self.q3_b = 7.35e+11
    self.q3_c = -2.82e-06
    self.q3_d = -3.28e+09
    self.q3_e = 1.27e-03
    # Break-in
    self.q4_a = 5.82e-01
    self.q4_SOC_mu    = 0.5
    self.q4_SOC_sigma = 2.09e-01
    self.q4_SOC_xi    = 5.83e-02
    self.q4_DOD_mu    = 0.5
    self.q4_DOD_sigma = 1.16e+00
    self.q4_DOD_xi    = -3.81e+00
    self.q4_b  = 2.54e+01
    self.q5    = 3.04e-03
    self.q6    = 1.44e+00
    # Long-term
    self.q7_a = -6.81e-06
    self.q7_b = 0.0
    self.q7_c = 2.12e-06
    self.q8   = 1.13
    # State
    self.gasper_cal_loss = 0.0
    self.gasper_bi_loss = 0.0
    self.gasper_long_loss = 0.0
    self.gasper_EFC_total = 0.0


def _Ua_from_SOC(self, soc):
    # Calculates anode potential Ua from SOC. 
    #Implements Appendix Eq. A-1 and A-2[cite: 636, 646].
    
    soc = min(1.0, max(0.0, float(soc)))
    # Anode stoichiometry xa from SOC (Eq. A-2)
    xa = 0.0085 + soc * (0.78 - 0.0085)

    # Graphite anode OCV Ua from xa (Eq. A-1)
    term1 = 0.6379
    term2 = 0.5416 * np.exp(-305.5309 * xa)
    term3 = 0.044 * np.tanh((xa - 0.1958) / -0.1088)
    term4 = -0.1978 * np.tanh((xa - 1.0571) / 0.0854)
    term5 = -0.6875 * np.tanh((xa + 0.0117) / 0.0529)
    term6 = -0.0175 * np.tanh((xa - 0.5692) / 0.0875)
    
    return term1 + term2 + term3 + term4 + term5 + term6

def _sig_y(self, amp, k, p, x):
    x = max(0.0, x)
    # Implements the core sigmoidal trajectory from Eq. 6 [cite: 404]
    # Note: The paper's formula is 2*q1*[1/2 - 1/(1+exp(...))], which simplifies
    # to q1 * (1 - 2/(1+exp(...))). The code's form is equivalent.
    return amp * (1.0 - 2.0 / (1.0 + np.exp((k * x)**p)))

def _inv_sig_y(self, amp, k, p, y):
    """
    Calculates the inverse of the sigmoid function to find the "virtual" time/EFC.
    This version is stabilized to prevent errors with np.log().
    """
    # Ensure y (current loss) is within the valid range [0, amp)
    y = min(max(y, 0.0), amp * (1.0 - 1e-12))
    if amp < 1e-12:
        return 0.0

    # Invert the function: y/amp = (e^arg - 1)/(e^arg + 1) => e^arg = (amp + y) / (amp - y)
    ratio = (amp + y) / (amp - y)
    
    # This check prevents np.log(negative)
    if ratio <= 0:
        return 0.0
    
    arg = np.log(ratio)
    x = (max(arg, 0.0))**(1.0 / max(p, 1e-12)) / max(k, 1e-12)
    return x
   
def _phi_skew(self, x, mu, sigma, xi):
    # Implements the skew-normal PDF from Eq. 13 [cite: 493]
    z = (x - mu) / max(sigma, 1e-12)
    # PDF part
    phi = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z * z)
    # CDF part (using error function)
    Phi = 0.5 * (1.0 + erf(xi * z / np.sqrt(2)))
    return (2.0 / max(sigma, 1e-12)) * phi * Phi

def _q1(self, T, soc):
    """ Implements calendar parameter q1 from Eq. 7 """
    Ua = _Ua_from_SOC(self, soc)
    T_safe = max(T, 1e-12)
    
    exp_term_b = np.exp(self.q1_b * (Ua**0.5) / (T_safe**2))
    exp_term_c = np.exp(self.q1_c * (Ua**0.5) / T_safe)
    
    return self.q1_a * exp_term_b * exp_term_c

def _q3(self, T, soc):
    """ Implements calendar parameter q3 from Eq. 8 """
    Ua = _Ua_from_SOC(self, soc)
    T_safe = max(T, 1e-12)

    exp_term_b = np.exp(self.q3_b * (Ua**(1/3)) / (T_safe**4))
    exp_term_c = np.exp(self.q3_c * (T_safe**3) * (Ua**0.25))
    exp_term_d = np.exp(self.q3_d * (Ua**(1/3)) / (T_safe**3))
    exp_term_e = np.exp(self.q3_e * (T_safe**2) * (Ua**0.25))
    
    return self.q3_a * exp_term_b * exp_term_c * exp_term_d * exp_term_e

def _q4_amp(self, soc, dod):
    """ Implements break-in amplitude q4 from Eq. 12 """
    # Skew-normal distributions for SOC and DOD
    phi_soc = _phi_skew(self, soc, self.q4_SOC_mu, self.q4_SOC_sigma, self.q4_SOC_xi)
    phi_dod = _phi_skew(self, dod, self.q4_DOD_mu, self.q4_DOD_sigma, self.q4_DOD_xi)
    
    # Sigmoid function of DOD
    sigmoid_dod = (2 * (0.5 - 1 / (1 + np.exp(self.q4_b * dod))))
    
    # Combine terms as per Eq. 12
    amp = self.q4_a * phi_soc * phi_dod * sigmoid_dod
    return max(0.0, amp)

def _q7(self, dod, crate):
    """ Implements long-term cycling rate q7 from Eq. 15 """
    # Note: Using crate**1 as per the equation, not crate**3.
    # Assuming q7_b is 0 as it's omitted from Table IV.
    #term_b = self.q7_b * (dod**2)
    term_b = self.q7_b * dod
    #term_c = self.q7_c * np.exp(dod * crate)
    term_c = self.q7_c * np.exp(dod**2 * crate**3)
    return max(self.q7_a + term_b + term_c, 1e-12)

def update_gasper(self, charge_power, discharge_power, include_float_degradation=True, dt_hours=1.0, temperature=298.15):
    # Rainflow cycles from SOC history
    active_now = (charge_power > 0) or (discharge_power > 0)

    T = temperature
    soc_now = min(1.0, max(0.0, self.ch / (self.C + 1e-12)))

    # === Calendar: virtual-time update in DAYS (Correct) ===
    d_cal = 0.0
    if include_float_degradation:
        q1 = max(1e-12, _q1(self, T, soc_now))
        q3 = max(1e-12, _q3(self, T, soc_now))

        t_prev_star_days = _inv_sig_y(self, q1, self.q2, q3, self.gasper_cal_loss)
        t_new_star_days  = t_prev_star_days + dt_hours / 24.0
        cal_new = _sig_y(self, q1, self.q2, q3, t_new_star_days)
        d_cal = max(0.0, cal_new - self.gasper_cal_loss)
        self.gasper_cal_loss = max(self.gasper_cal_loss, cal_new)

    # === Break-in and Long-term: per detected cycle (Updated) ===
    d_bi = 0.0
    d_long = 0.0

    #dods = [0.0]
    #ahrss = [0.0]
    #crates = [0.0]
    #q7s = [0.0]
    bi_prev_star = 0
    cycles = self.cycle_check_rainflow(dt_hours, active_now=active_now)
    if cycles:
        for cyc in cycles:

            dod   = max(0.0, min(1.0, cyc["DOC"]))
            ahrs  = max(1e-9, float(cyc["active_hours"]))
            if ahrs <= 1e-6:
                ahrs = max(1e-6, float(cyc["hours"]))

            # Average C-rate during active segments; keep within identification range
            crate = min(2.0, max(0.0, 2.0 * dod / ahrs))

            # EFC increment: for a closed cycle, EFC = count * DOD
            dEFC  = max(0.0, cyc["count"] * dod)

            # ---- PATCH: apply break-in only during high-use periods (≈≥ 2 EFC/day) ----
            efc_per_day = (dEFC / ahrs) * 24.0  # simple local rate proxy over the cycle duration
            q4 = _q4_amp(self, cyc["soc_mean"], dod)
            if (q4 > 0.0) and (efc_per_day >= 2.0):
                bi_prev_star = _inv_sig_y(self, q4, self.q5, self.q6, self.gasper_bi_loss)
                bi_new = _sig_y(self, q4, self.q5, self.q6, bi_prev_star + dEFC)
                d_bi_step = max(0.0, bi_new - self.gasper_bi_loss)
                self.gasper_bi_loss = max(self.gasper_bi_loss, bi_new)
                d_bi += d_bi_step
            # ---- END PATCH ----

            # Long-term cycling is applied for all cycles
            q7 = _q7(self, dod, crate)
            #long_prev_star = 0.0 if self.gasper_long_loss <= 0 else (self.gasper_long_loss / max(q7, 1e-12))**(1.0 / max(self.q8, 1e-12))
            long_prev_star = 0.0 if self.gasper_long_loss <= 0 else (self.gasper_long_loss**(1.0 / self.q8) / max(q7, 1e-12))
            #long_new = max(0.0, q7) * (long_prev_star + dEFC)**self.q8
            long_new = (max(0.0, q7) * (long_prev_star + dEFC))**self.q8
            d_long_step = max(0.0, long_new - self.gasper_long_loss)
            self.gasper_long_loss = max(self.gasper_long_loss, long_new)
            d_long += d_long_step

            self.gasper_EFC_total += dEFC

    # Apply total loss and update history
    if self.include_degradation:
        current_total_loss = self.gasper_cal_loss + self.gasper_bi_loss + self.gasper_long_loss
        self.C = max(0.0, self.C_init * (1.0 - current_total_loss))
        MAX_C = self.C if not self.rule_80_20 else 0.8*self.C
        self.ch = min(self.ch, MAX_C)

    # History logging for diagnostics
    d_loss_step = d_cal + d_bi + d_long
    self.aging_factor_float_history = np.append(self.aging_factor_float_history, d_cal)
    self.aging_factor_cyclic_history = np.append(self.aging_factor_cyclic_history, d_bi + d_long)
    self.aging_factor_history = np.append(self.aging_factor_history, d_loss_step)

    if len(self.capacity_history) > 0:
        delta_SOH = (self.capacity_history[-1] - self.C)/(max(self.C_init,1e-30)*(1-self.END_OF_LIFE))
        self.deltaSOH_history = np.append(self.deltaSOH_history, delta_SOH)

        total_step_loss = d_cal + d_bi + d_long
        if total_step_loss > 1e-12:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, delta_SOH * (d_cal / total_step_loss))
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, delta_SOH * ((d_bi + d_long) / total_step_loss))
        else:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, 0.0)
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, 0.0)
