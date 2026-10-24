"""
Magnor Model functions

Source:

D. Magnor, J. B. Gerschler, M. Ecker, P. Merk, D. U. Sauer,
Concept of a battery aging model for lithium-ion batteries considering the lifetime dependency on the operation strategy,
in: Proceedings of the European Photovoltaic Solar Energy Conference, Hamburg, Germany, 2009, pp. 21–25
URL https://www.researchgate.net/publication/293145570_CONCEPT_OF_A_BATTERY_AGING_MODEL_FOR_LITHIUM-ION_BATTERIES_CONSIDERING_THE_LIFETIME_DEPENDENCY_ON_THE_OPERATION_STRATEGY
"""

import numpy as np
from collections import namedtuple
Cycle = namedtuple("Cycle", ["delta_soc", "mean_soc", "count", "t_start", "t_end", "kind"])

def init_magnor(self):
    self.a = 1.2698e6
    self.b = -1.3133
    self.d = 2.0
    self.e = -1.2
    self.f = -0.0275
    self.Ea = 25000.0
    self.LIFETIME = 15  # Years
    self.deltaT = 15
    
    # Magnor interval aggregation 
    self.interval_hours = 24.0
    self._acc_float = 0.0
    self._acc_cyc = 0.0
    self._acc_elapsed = 0.0
    self.consumed_life = 0.0
    self.SOH = 1.0
    
    # Buffers for Magnor cycle detector path ---
    self._magnor_t = []                 # absolute time [h], monotone
    self._magnor_soc = []               # SOC in [0..1]
    self._magnor_elapsed_h_total = 0.0  # running absolute time [h]
    self._magnor_damage_counted = 0.0   # cumulative cycle damage already accounted
    self._last_dt_hours = 0.0           # step size cache

    
    # Linearized cyclic degradation params (see 'degradation_efficiency_curve.ipynb')
    self.cyclic_degradation_curve_lin_slopes = [0.00015675769, 0.00024698501, 0.00033011383, 0.00040564024]  # Values obtained when y-values multiplied by 100 (in the unit of %)
    self.cyclic_degradation_curve_lin_inters = [0.0, -0.00081204588, -0.002807137746, -0.0072631956]  # Values obtained when y-values multiplied by 100 (in the unit of %)
    self.cyclic_degradation_curve_lin_SOCS = [0, 10, 25, 60]  # Starting points for change-points
    # Linearized float degradation params (see 'degradation_efficiency_curve.ipynb')
    self.float_degradation_curve_lin_slopes = [0.0008390540006103, 0.0023000485806749594, 0.007772392380464037, 0.025749092486947102] # Values represent SOC-aging, not float aging,
                                                                                                                                      # divide these with 10*8760 to get the float params
    self.float_degradation_curve_lin_inters = [0.5199433185834548, 0.48487944866190297, 0.16201116447434732, -1.34803164447023] # Values represent SOC-aging, not float aging,
                                                                                                                                # divide these with 10*8760 to get the float params
    self.float_degradation_curve_lin_SOCS = [0, 25, 60, 85]  # Starting points for change-points
    
    
def update_magnor(self, charge_power, discharge_power, include_float_degradation=True, dt_hours=1.0, temperature=298.15):
    
    detection_type = "rainflow" # Magnor_detector, rainflow or switch
    """
    detection_type:
        "switch"   -> uses self.cycle_check(...) from Battery.py
        "rainflow" -> uses self.cycle_check_rainflow(...) from Battery.py
        "Magnor_detector" -> uses Magnor macro+micro detector defined below(...)
    """
    # Cache dt for detector bookkeeping
    self._last_dt_hours = float(dt_hours)
    
    # === CYCLIC aging ===
    if detection_type == "switch":
        aging_factor_cyc = 0.0
        flip = self.cycle_check(charge_power, discharge_power, dt_hours)  # defined in Battery.py
        if flip.get("completed", False):
            DOC = max(0.0, float(flip.get("DOC", 0.0)))
            if DOC > 0.0:
                Nmax = self.a * (100.0 * DOC) ** self.b
                aging_factor_cyc = 1.0 / max(Nmax, 1e-30)

    elif detection_type == "rainflow":
        aging_factor_cyc = 0.0
        cycles = self.cycle_check_rainflow(dt_hours)
        if cycles:
            for cyc in cycles:
                DOC = cyc["DOC"]
                if DOC > 0.0:
                    Nmax = self.a * (100.0 * DOC)**self.b
                    aging_factor_cyc += cyc["count"] / max(Nmax, 1e-30)
                
    elif detection_type == "Magnor_detector":
        # Replacement path: macro+micro Magnor detector
        damage_inc = _magnor_cycles_increment(self)
        aging_factor_cyc = damage_inc / max(dt_hours, 1e-30)

    else:
        raise ValueError("detection_type must be 'switch' or 'rainflow'")
            
    # === FLOAT aging — SOC-exponential at reference temperature
    if include_float_degradation:
        SOCp = 100.0 * (self.ch / (self.C + 1e-10))          # SOC in %
        soc_term = 1.0 / (self.d + self.e*np.exp(self.f*(100.0 - SOCp)))
        
        #kT = 1
        # Since T=Tref, the Arrhenius term exp(...) is 1. kT is 1.
        #kT = np.exp(-self.Ea/self.R * (1.0/temperature - 1.0/self.Tref))
        kT = 2**((temperature-self.Tref)/self.deltaT)
        per_hour = soc_term * kT / (self.LIFETIME * 8760.0)       # ref T by design
        aging_factor_float = per_hour * dt_hours
    else:
        aging_factor_float = 0.0        

    # Accumulate inside the current interval
    self._acc_float += aging_factor_float
    self._acc_cyc   += aging_factor_cyc
    self._acc_elapsed += dt_hours

    # Per-step diagnostics (optional): store raw partials
    self.aging_factor_float_history = np.append(self.aging_factor_float_history, aging_factor_float)
    self.aging_factor_cyclic_history = np.append(self.aging_factor_cyclic_history, aging_factor_cyc)

    # Finalize at interval boundary
    if self._acc_elapsed + 1e-12 >= self.interval_hours:
        resulting = max(self._acc_float, self._acc_cyc)   # Magnor rule
        self.consumed_life += resulting                    # Σ over periods
        self.aging_factor_history = np.append(self.aging_factor_history, resulting)

        # Linear SOH and capacity update per Magnor
        #self.SOH = max(0.0, 1.0 - self.consumed_life)     # SOH goes 1→0
        self.SOH = 1.0 - self.consumed_life
        if self.include_degradation:
            # Map SOH linearly to capacity from 1.0 to END_OF_LIFE at SOH=0
            cap_frac = self.END_OF_LIFE + (1 - self.END_OF_LIFE) * self.SOH
            self.C = self.C_init * cap_frac
            MAX_C = self.C if not self.rule_80_20 else 0.8*self.C
            self.ch = min(self.ch, MAX_C)

        delta_SOH = (self.capacity_history[-1] - self.C)/(max(self.C_init,1e-30)*(1-self.END_OF_LIFE))
        self.deltaSOH_history = np.append(self.deltaSOH_history, delta_SOH)

        # Attribute the SOH drop to the dominating mechanism for this interval
        if self._acc_float >= self._acc_cyc:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, delta_SOH)
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, 0.0)
        else:
            self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, 0.0)
            self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, delta_SOH)

        # reset interval
        self._acc_float = 0.0
        self._acc_cyc   = 0.0
        self._acc_elapsed = 0.0
    else:
        # keep series lengths aligned when not finalizing this step
        self.aging_factor_history = np.append(self.aging_factor_history, 0.0)
        self.deltaSOH_history = np.append(self.deltaSOH_history, 0.0)
        self.deltaSOH_float_history = np.append(self.deltaSOH_float_history, 0.0)
        self.deltaSOH_cyclic_history = np.append(self.deltaSOH_cyclic_history, 0.0)



# =========================
# Internal: incremental cycle damage via Magnor detector
# =========================
def detect_cycles_magnor(t, soc,
                         macro_ma_window_s=3600.0,
                         macro_tolerance_band=0.02,
                         min_delta_soc=1e-4):
    """
    Detect micro and macro cycles for a Magnor-style lifetime model.

    Parameters
    ----------
    t : array_like, shape (N,)
        Monotonic time stamps [s].
    soc : array_like, shape (N,)
        State of charge trace in [0..1].
    macro_ma_window_s : float
        Moving-average window for macro-profile smoothing [s].
        Use ~0.5–2 h for EV duty or per your paper’s intervalization.
    macro_tolerance_band : float
        Deadband (absolute SOC) used to simplify the macro profile (±band).
        Typical 0.01–0.03 for energy cells; tune to your validation.
    min_delta_soc : float
        Minimum ΔSOC to accept a cycle after rainflow (filters numerical crumbs).

    Returns
    -------
    cycles : list[Cycle]
        Combined list of micro and macro cycles. Each has:
        - delta_soc : absolute range of the cycle (peak-to-valley) in SOC units
        - mean_soc  : midpoint of the cycle in SOC units
        - count     : 1.0 for closed cycles, 0.5 for residue
        - t_start   : approximate start time [s] (valley/peak index)
        - t_end     : approximate end time [s] (closing valley/peak index)
        - kind      : "micro" or "macro"
    """
    t = np.asarray(t, dtype=float)
    soc = np.asarray(soc, dtype=float)
    if t.ndim != 1 or soc.ndim != 1 or t.size != soc.size or t.size < 3:
        raise ValueError("t and soc must be 1D arrays of equal length >= 3")

    # 1) Build micro profile: turning points from raw SOC
    tp_idx_micro = _turning_points(soc)
    t_micro = t[tp_idx_micro]
    soc_micro = soc[tp_idx_micro]

    # 2) Build macro profile:
    # 2a) moving-average smoothing by time window
    k = max(1, int(round(_window_len_from_seconds(t, macro_ma_window_s))))
    soc_ma = _moving_average_irregular(t, soc, k)
    # 2b) tolerance-band simplification (deadband tracker)
    tp_idx_macro = _tolerance_band_simplify(t, soc_ma, band=macro_tolerance_band)
    t_macro = t[tp_idx_macro]
    soc_macro = soc_ma[tp_idx_macro]

    # 3) Rainflow on both profiles
    cycles_micro = _rainflow_cycles(t_micro, soc_micro, min_delta_soc, kind="micro")
    cycles_macro = _rainflow_cycles(t_macro, soc_macro, min_delta_soc, kind="macro")

    # 4) Combine
    return cycles_micro + cycles_macro


def _window_len_from_seconds(t, window_s):
    # Approximate samples per window using median dt, protect against garbage
    dt = np.diff(t)
    mdt = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    return max(1.0, window_s / mdt)


def _moving_average_irregular(t, x, k):
    # Uniform-length trailing window MA using integer k; cheap and cheerful
    k = int(max(1, k))
    if k == 1:
        return x.copy()
    # Use cumulative sum for speed
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[k:] - c[:-k]) / k
    # Align length by padding the first k-1 with the first computed value
    pad = np.full(k - 1, out[0])
    return np.concatenate([pad, out])


def _turning_points(y):
    y = np.asarray(y)
    n = y.size
    if n < 3:
        return np.arange(n)
    idx = [0]
    for i in range(1, n - 1):
        if (y[i] > y[i - 1] and y[i] >= y[i + 1]) or (y[i] < y[i - 1] and y[i] <= y[i + 1]):
            # keep only if value changed vs last turning point
            if y[i] != y[idx[-1]]:
                idx.append(i)
        if y[i] == y[i - 1] == y[i + 1]:
            continue
    if idx[-1] != n - 1:
        idx.append(n - 1)
    # remove successive duplicates by value
    clean = [idx[0]]
    for j in idx[1:]:
        if y[j] != y[clean[-1]]:
            clean.append(j)
    return np.array(clean, dtype=int)


def _tolerance_band_simplify(t, y, band=0.02):
    t = np.asarray(t)
    y = np.asarray(y)
    n = y.size
    if n < 3:
        return np.arange(n)

    anchors = [0]
    last_anchor = 0
    last_dir = 0
    upper = y[last_anchor] + band
    lower = y[last_anchor] - band

    for i in range(1, n):
        val = y[i]
        step_dir = np.sign(val - y[i - 1])
        if step_dir != 0 and step_dir != last_dir:
            anchors.append(i - 1)
            last_anchor = anchors[-1]
            upper = y[last_anchor] + band
            lower = y[last_anchor] - band
            last_dir = step_dir
        if val > upper or val < lower:
            anchors.append(i)
            last_anchor = i
            upper = y[last_anchor] + band
            lower = y[last_anchor] - band
            last_dir = step_dir if step_dir != 0 else last_dir

    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    anchors = np.unique(anchors)
    # return absolute indices, not indices into y[anchors]
    tp_rel = _turning_points(y[anchors])
    return anchors[tp_rel]


def _rainflow_cycles(t, y, min_delta_soc, kind):
    """
    ASTM E1049-85 four-point algorithm returning closed and residue cycles.
    Returns list[Cycle] with delta_soc = range (peak-valley), count = 1 or 0.5.
    """
    y = np.asarray(y)
    t = np.asarray(t)
    n = y.size
    if n < 2:
        return []

    # Step 1: Extract turning points (defensive)
    tp = _turning_points(y)
    y = y[tp]
    t = t[tp]

    cycles = []

    def make_cycle(i1, i2, i3, i4, full=True):
        # cycle from y2 to y3 closes against y1/y4
        y2, y3 = y[i2], y[i3]
        delta = abs(y3 - y2)  # full range
        if delta < min_delta_soc:
            return
        mean = 0.5 * (y3 + y2)
        # crude timing: start at i2, end at i3 in the turning-point sequence
        cycles.append(Cycle(delta_soc=delta,
                            mean_soc=mean,
                            count=1.0 if full else 0.5,
                            t_start=t[i2],
                            t_end=t[i3],
                            kind=kind))

    # ASTM-like implementation
    X = y.tolist()
    I = list(range(len(X)))
    s = []
    idx = []
    for i, xi in enumerate(X):
        s.append(xi)
        idx.append(I[i])
        while len(s) >= 4:
            S1 = abs(s[-1] - s[-2])
            S2 = abs(s[-2] - s[-3])
            S3 = abs(s[-3] - s[-4])
            if S2 <= S1 and S2 <= S3:
                # Close a cycle around s[-3]..s[-2]
                i2, i3 = idx[-3], idx[-2]
                # Parent points i1, i4 not used for timing here; range from i2..i3
                make_cycle(0, i2, i3, 0, full=True)
                # Remove middle pair s[-3], s[-2]
                del s[-3:-1]
                del idx[-3:-1]
            else:
                break

    # Residue: remaining ranges become half-cycles
    for j in range(len(s) - 1):
        i2, i3 = idx[j], idx[j + 1]
        # Half-cycle
        delta = abs(y[i3] - y[i2])
        if delta >= min_delta_soc:
            mean = 0.5 * (y[i3] + y[i2])
            cycles.append(Cycle(delta_soc=delta,
                                mean_soc=mean,
                                count=0.5,
                                t_start=t[i2],
                                t_end=t[i3],
                                kind=kind))
    return cycles


def _magnor_cycles_increment(self):
    # Append current point
    soc_now = self.ch / max(self.C, 1e-30)
    self._last_dt_hours = float(self._last_dt_hours)
    self._magnor_elapsed_h_total += self._last_dt_hours
    self._magnor_t.append(self._magnor_elapsed_h_total)
    self._magnor_soc.append(soc_now)

    # only compute cycles when we’re about to finalize the interval
    if self._acc_elapsed + self._last_dt_hours < self.interval_hours - 1e-12:
        return 0.0

    if len(self._magnor_t) < 4:
        return 0.0

    # Detect macro + micro cycles over the full trace
    cycles = detect_cycles_magnor(
        t=np.asarray(self._magnor_t, dtype=float) * 3600.0,  # seconds
        soc=np.asarray(self._magnor_soc, dtype=float),
        macro_ma_window_s=3600.0,        # 1-hour smoothing
        macro_tolerance_band=0.02,       # ±2% SOC band
        min_delta_soc=1e-4
    )

    # Total damage so far
    total_damage = 0.0
    for cyc in cycles:
        DOC = float(cyc.delta_soc)  # 0..1
        if DOC <= 0.0:
            continue
        Nmax = self.a * (100.0 * DOC) ** self.b
        total_damage += float(cyc.count) / max(Nmax, 1e-30)

    # Increment since last time
    new_damage = max(0.0, total_damage - self._magnor_damage_counted)
    self._magnor_damage_counted = total_damage
    return new_damage

