# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:56:38 2026
Latest modification on Fri Oct 23 2026 by ppkart
@authors: idafu and ppkart
"""

import pandas as pd
import numpy as np
import pickle
import sys, os
import tqdm

sys.path.append(os.path.dirname(os.path.join('Battery')))
from Battery.Battery import Battery

# ============================================================
# Global settings
# ============================================================
#DT = 0.25                   # hours per timestep (15 min)
DT = 1

# ============================================================
# Helper functions
# ============================================================
def proportional_allocate(target, caps, weights):
    """
    Allocate a target quantity proportionally among participants.

    Parameters
    ----------
    target : float
        Total amount to allocate.

    caps : ndarray
        Maximum allocation capacity of each participant.

    weights : ndarray
        Relative allocation weights.

    Returns
    -------
    ndarray
        Allocation for each participant.
    """
    caps = np.asarray(caps, float)
    weights = np.asarray(weights, float)
    alloc = np.zeros_like(caps)
    if target <= 1e-12:
        return alloc
    active = np.where((caps > 1e-12) & (weights > 0))[0]
    remaining = float(target)
    while remaining > 1e-12 and len(active) > 0:
        W = weights[active].sum()
        if W <= 0:
            break
        share = remaining * (weights[active] / W)
        limited = np.minimum(share, caps[active])
        alloc[active] += limited
        remaining -= limited.sum()
        caps[active] -= limited
        active = active[caps[active] > 1e-12]
    return alloc


def discharge_limits(residual_W, soc_Wh, battP_W, eff, invP_W, dt, battery):
    """
    Calculate feasible battery discharge power.

    Parameters
    ----------
    residual_W : float
        Remaining load demand.

    soc_Wh : float
        Battery energy content.

    battP_W : float
        Battery power limit.

    invP_W : float
        Inverter power limit.

    dt : float
        Simulation timestep [h].

    battery : Battery

    Returns
    -------
    tuple
        (discharge_power_W, remaining_soc_Wh)
    """
    if residual_W <= 0:
        return 0.0, soc_Wh
    required = min(residual_W, battP_W, invP_W)
    energy_needed_Wh = required * dt / eff
    actual_energy_Wh = min(soc_Wh, energy_needed_Wh)
    actual_power_W = actual_energy_Wh * eff / dt
    
    return actual_power_W, soc_Wh - actual_energy_Wh


def get_indoor_temperature(air_temp, dem, tau_hours=48, dt_hours=1, Tset=20.0):
    """
    Simulate cabin indoor temperatures using a first-order
    RC thermal model.

    Assumptions
    -----------
    - Occupied cabin:
      Indoor temperature fixed at Tset.

    - Unoccupied cabin:
      Indoor temperature follows an exponential decay
      toward ambient temperature.

    Parameters
    ----------
    air_temp : pd.Series
        Outdoor temperature [°C]

    dem : ndarray, shape (n_timesteps, n_profiles)
        Electricity demand profiles. Demand > 0 => occupied/heated.
    
    tau_hours : float
        Thermal time constant of cabin [h]
    
    dt_hours : float
        Time step [h]
    
    Tset : float
        Heating setpoint [°C]

    Returns
    -------
    Tin : ndarray, shape (n_timesteps, n_profiles)
        Simulated indoor temperatures [°C]
    """

    n_steps = len(air_temp)
    n_profiles = dem.shape[1]

    dem = dem.to_numpy()

    alpha = 1 - np.exp(-dt_hours / tau_hours)

    Tin = np.zeros((n_steps, n_profiles))

    # Initial condition
    Tin[0, :] = air_temp.iloc[0]

    tout_values = air_temp.to_numpy()

    for p in range(n_profiles):
        occupied = dem[:, p] > 0
        for t in range(1, n_steps):
            # Passive RC response
            trc = (Tin[t - 1, p] + alpha * (tout_values[t - 1] - Tin[t - 1, p]))[0]
            
            # Heating when occupied
            if occupied[t]:
                Tin[t, p] = Tset
            else:
                Tin[t, p] = trc

    return Tin


def charge_limits(surplus_W, soc_Wh, cap_Wh, battP_W, eff, invP_W, dt, battery):
    """
    Calculate feasible battery charging power.

    Parameters
    ----------
    surplus_W : float
        Surplus PV generation.

    soc_Wh : float
        Battery energy content.

    cap_Wh : float
        Battery capacity.

    battP_W : float
        Battery power limit.

    invP_W : float
        Inverter power limit.

    dt : float
        Simulation timestep [h].

    battery : Battery

    Returns
    -------
    tuple
        (charge_power_W, resulting_soc_Wh)
    """

    if surplus_W <= 0:
        return 0.0, soc_Wh
    allowed_W = min(surplus_W, battP_W, invP_W)
    cap_headroom_W = max((cap_Wh - soc_Wh) / (dt * eff), 0.0)
    allowed_W = min(allowed_W, cap_headroom_W)
    if allowed_W <= 0:
        return 0.0, soc_Wh
    store_Wh = allowed_W * dt * eff
    
    return allowed_W, min(soc_Wh + store_Wh, cap_Wh)

# ============================================================
# 1) Off-grid model
# ============================================================
def simulate_offgrid(dem_W, pv_W, indoor_temperature,
                     battery_capacity, battery_startSOC,
                     battery_maxpower, battery_eff, inverter_maxpower,
                     degradation_model, dt=DT, 
                     include_degradation=False, idle_soc_limit=1):
    
    """
    Simulate independent off-grid cabin operation.

    Dispatch sequence
    -----------------
    1. PV serves local load.
    2. Battery serves remaining load.
    3. Unserved demand is recorded as lost load.
    4. Remaining PV charges the battery.
    5. Excess PV is curtailed.

    Parameters
    ----------
    dem_W : ndarray of shape (T, N)
        Electrical demand profiles for each cabin [W].

        - T = number of simulation timesteps
        - N = number of cabins

    pv_W : ndarray of shape (T, N)
        PV generation profiles for each cabin [W].

    indoor_temperature : ndarray of shape (T, N)
        Indoor temperature profile for each cabin [°C].

    battery_capacity : ndarray of shape (N,)
        Initial battery capacity of each cabin [Wh].

    battery_startSOC : ndarray of shape (N,)
        Initial battery state of charge as a fraction
        between 0 and 1.

    battery_maxpower : ndarray of shape (N,)
        Maximum charging and discharging power of each
        battery [W].

    battery_eff : ndarray of shape (N,)
        Round-trip or operating battery efficiency used in
        charge/discharge calculations [-].

        Typical values are between 0.85 and 1.0.

    inverter_maxpower : ndarray of shape (N,)
        Maximum inverter power rating [W].

        Battery charge and discharge power cannot exceed
        this limit.

    degradation_model : str
        Name of the battery degradation model to use.

        Examples:
            - "Naumann"
            - "Gasper"
            - "Magnor"

    dt : float, optional
        Simulation timestep length [h].

        Default is globally defined ``DT``.

    include_degradation : bool, optional
        If True, battery degradation, capacity fade and
        self-discharge are simulated.

        If False, battery capacity remains constant.

    idle_soc_limit : float, optional
        Maximum allowed state of charge during predicted
        idle periods.

        The strategy attempts to reduce time spent at high
        SOC when the cabin is expected to remain unused,
        while still ensuring the battery can be fully charged
        before the next expected visit.

        Examples:
            1.0 = no SOC limitation
            0.8 = maintain approximately 80 % SOC
            0.5 = maintain approximately 50 % SOC

    Returns
    -------
    dict
        Simulation results including energy balances,
        battery state of charge, capacity degradation,
        self-discharge and curtailed PV.
    """

    T, N = dem_W.shape   # T=simulation steps, N=number of loads 
    soc = np.zeros((T, N))
    soc[0, :] = battery_startSOC * battery_capacity
    lost = np.zeros((T, N))
    served = np.zeros((T, N))
    soc_battery_class = np.zeros((T, N))
    capacity = np.zeros((T, N))
    wasted_pv = np.zeros((T, N))
    self_discharges = np.zeros((T, N))

    # Initialize battery classes for each load
    batteries = [Battery(include_degradation=include_degradation, 
                         model=degradation_model, capacity=battery_capacity[i], rule_80_20=False,
                         specific_price=0, battery_max_Crate=1) for i in range(N)]
    for i in range(N):
        batteries[i].MAX_CH_POWER = battery_maxpower[i]
        batteries[i].MAX_DCH_POWER = battery_maxpower[i]
        batteries[i].ch = battery_startSOC[i] * batteries[i].C_init
        # Initialize first self-dishcarge
        batteries[i].self_discharge_history = np.array([0])

    #for t in range(1, T):
    for t in tqdm.tqdm(range(1, T), desc=f"{degradation_model} - {int(idle_soc_limit*100)}%"):
        d = dem_W[t, :]  # Demands for given hour
        p = pv_W[t, :]  # Productions for given hour
        soc_prev = soc[t-1, :].copy()

        # Limit idle SOC if the next day would be unused. Then start charging battery with perfect 
        # solar production forecast so that it's full when the owner arrives there
        next_24h_load = dem_W[t:t+24, :].sum(axis=0)
        pv_cumsum = np.cumsum(pv_W[t:, :], axis=0)
        steps_until_battery_is_full = np.argmax(pv_cumsum >= np.array([(batteries[0].C-batteries[0].ch)/battery_eff[0], 
                                                                       (batteries[1].C-batteries[1].ch)/battery_eff[1], 
                                                                       (batteries[2].C-batteries[2].ch)/battery_eff[2],
                                                                       (batteries[3].C-batteries[3].ch)/battery_eff[3]]),
                                                                       axis=0)
        steps_until_next_cabin_visit = np.argmax(dem_W[t:, :] > np.array([0, 0, 0, 0]), axis=0)
        
        # Direct PV to own load
        direct = np.minimum(d, p)
        residual = d - direct

        # Battery discharge
        dis_W = np.zeros(N)
        soc_mid = soc_prev.copy()
        for i in range(N):
            idle_soc_limit_discharge = 0
            if (next_24h_load[i] == 0) and (steps_until_battery_is_full[i] < steps_until_next_cabin_visit[i]):
                idle_soc_limit_discharge = max((batteries[i].ch - batteries[i].C_init*idle_soc_limit)*battery_eff[i], 0)

            discharge_need = max(residual[i], idle_soc_limit_discharge)
            dch_w, soc_mid[i] = discharge_limits(
                discharge_need, soc_prev[i],
                battery_maxpower[i], battery_eff[i],
                inverter_maxpower[i], dt, batteries[i]
            )
            dis_W[i] = dch_w

        unmet = np.maximum(residual - dis_W, 0.0)

        # Charge battery with PV surplus
        char_W = np.zeros(N)
        surplus = p - direct
        soc_after = soc_mid.copy()

        for i in range(N):
            if (next_24h_load[i] == 0) and (steps_until_battery_is_full[i] < steps_until_next_cabin_visit[i]):
                surplus[i] = max((batteries[i].C_init*idle_soc_limit - batteries[i].ch)/battery_eff[i], 0)
            
            ch_w, soc_after[i] = charge_limits(
                surplus[i], soc_mid[i],
                batteries[i].C, battery_maxpower[i],
                battery_eff[i], inverter_maxpower[i], dt, batteries[i]
            )
            char_W[i] = ch_w
        
        wasted_pv[t, :] = surplus - char_W

        # Update batteries
        for i in range(N):
            batteries[i].update(charge_power=char_W[i], discharge_power=dis_W[i], dt_hours=dt, temp=indoor_temperature[t, i]+273.15)
            self_discharges[t, i] = batteries[i].self_discharge_history[t]
            soc_battery_class[t, i] = batteries[i].ch
            soc_after[i] = min(batteries[i].C, soc_after[i])
            capacity[t, i] = batteries[i].C

        served[t, :] = d - unmet
        lost[t, :] = unmet
        soc[t, :] = soc_after

    return {"soc_Wh": soc, 
            "lost_W": lost, 
            "served_W": served, 
            "soc_Wh_battery_class": soc_battery_class,
            "capacity": capacity,
            "load_W": dem_W,
            "production": pv_W,
            "wasted_PV_W": wasted_pv,
            "indoor_temperature": indoor_temperature,
            "self_discharge": self_discharges
            }

# ============================================================
# 2) Rule-based
# ============================================================
def simulate_rulebased(dem_W, pv_W, indoor_temperature,
                       battery_capacity, battery_startSOC,
                       battery_maxpower, battery_eff, inverter_maxpower,
                       degradation_model, dt=DT, 
                       include_degradation=False):
    """
    Simulate rule-based energy sharing between cabins.

    Energy dispatch sequence
    ------------------------
    1. Own PV → own load
    2. Own battery → own load
    3. Community PV → community loads
    4. Community battery discharge → community loads
    5. Own PV → own battery
    6. Community PV → community batteries

    The model prioritizes local self-consumption and then
    distributes surplus generation and available battery
    storage between cabins according to a proportional
    allocation scheme.

    Parameters
    ----------
    Parameters
    ----------
    dem_W : ndarray of shape (T, N)
        Electrical demand profiles for each cabin [W].

        - T = number of simulation timesteps
        - N = number of cabins

    pv_W : ndarray of shape (T, N)
        PV generation profiles for each cabin [W].

    indoor_temperature : ndarray of shape (T, N)
        Indoor temperature profile for each cabin [°C].

    battery_capacity : ndarray of shape (N,)
        Initial battery capacity of each cabin [Wh].

    battery_startSOC : ndarray of shape (N,)
        Initial battery state of charge as a fraction
        between 0 and 1.

    battery_maxpower : ndarray of shape (N,)
        Maximum charging and discharging power of each
        battery [W].

    battery_eff : ndarray of shape (N,)
        Round-trip or operating battery efficiency used in
        charge/discharge calculations [-].

        Typical values are between 0.85 and 1.0.

    inverter_maxpower : ndarray of shape (N,)
        Maximum inverter power rating [W].

        Battery charge and discharge power cannot exceed
        this limit.

    degradation_model : str
        Name of the battery degradation model to use.

        Examples:
            - "Naumann"
            - "Gasper"
            - "Magnor"

    dt : float, optional
        Simulation timestep length [h].

        Default is globally defined ``DT``.

    include_degradation : bool, optional
        If True, battery degradation, capacity fade and
        self-discharge are simulated.

        If False, battery capacity remains constant.

    Returns
    -------
    dict
        Dictionary containing simulation results and all
        tracked energy flows.

        Keys
        ----
        soc_Wh : ndarray
            Battery state of charge [Wh].

        lost_W : ndarray
            Unserved electrical demand [W].

        served_W_self : ndarray
            Demand supplied directly by local PV and the
            cabin's own battery [W].

        served_W_comm : ndarray
            Demand supplied through community energy sharing
            [W].

        comm_pv_to_load_to_others : ndarray
            PV energy exported from a cabin and consumed by
            other cabins' loads [W].

        comm_pv_to_load_to_self : ndarray
            Community PV energy received by a cabin to serve
            its own load [W].

        comm_pv_to_bat_to_others : ndarray
            PV energy exported from a cabin to charge other
            cabins' batteries [W].

        comm_pv_to_bat_to_self : ndarray
            Community PV energy received for charging the
            cabin's own battery [W].

        comm_bat_to_load_to_others : ndarray
            Battery energy discharged by a cabin and supplied
            to other cabins' loads [W].

        comm_bat_to_load_to_self : ndarray
            Battery energy received from community batteries
            to serve a cabin's load [W].

        disappeared : ndarray
            Energy removed when calculated SOC exceeds the
            current degraded battery capacity [Wh].

        soc_Wh_battery_class : ndarray
            State of charge tracked internally by the
            Battery class [Wh].

        capacity : ndarray
            Available battery capacity after degradation [Wh].

        load_W : ndarray
            Original demand profiles [W].

        production : ndarray
            Original PV generation profiles [W].

        indoor_temperature : ndarray
            Indoor temperature profiles used during the
            simulation [°C].

        self_dishcarge : ndarray
            Battery self-discharge losses [Wh].
    """

    T, N = dem_W.shape
    soc = np.zeros((T, N))
    soc[0, :] = battery_startSOC * battery_capacity
    lost = np.zeros((T, N))
    served_self = np.zeros((T, N))
    served_comm = np.zeros((T, N))
    comm_pv_to_load_to_others = np.zeros((T, N))
    comm_pv_to_load_to_self = np.zeros((T, N))
    comm_pv_to_bat_to_others = np.zeros((T, N))
    comm_pv_to_bat_to_self = np.zeros((T, N))
    comm_bat_to_load_to_others = np.zeros((T, N))
    comm_bat_to_load_to_self = np.zeros((T, N))
    disappeared = np.zeros((T, N))
    soc_battery_class = np.zeros((T, N))
    capacity = np.zeros((T, N))
    self_discharges = np.zeros((T, N))

    # Initialize battery classes for each load
    batteries = [Battery(include_degradation=include_degradation, 
                         model=degradation_model, capacity=battery_capacity[i], rule_80_20=False,
                         specific_price=0, battery_max_Crate=1) for i in range(N)]
    
    for i in range(N):
        batteries[i].MAX_CH_POWER = battery_maxpower[i]
        batteries[i].MAX_DCH_POWER = battery_maxpower[i]
        batteries[i].ch = battery_startSOC[i] * batteries[i].C_init
        batteries[i].previous_ch = battery_startSOC[i] * batteries[i].C_init
        # Initialize first self-dishcarge
        batteries[i].self_discharge_history = np.array([0])

    #for t in range(1, T):
    for t in tqdm.tqdm(range(1, T), desc=degradation_model):
        d = dem_W[t, :]
        p = pv_W[t, :]
        soc_prev = soc[t-1, :].copy()

        # Step 1: Direct PV to own load
        direct = np.minimum(d, p)
        residual = d - direct

        # Step 2: Own battery discharge
        dis_own_W = np.zeros(N)
        soc_mid = soc_prev.copy()
        for i in range(N):
            dch_w, soc_mid[i] = discharge_limits(
                residual[i], soc_prev[i],
                battery_maxpower[i], battery_eff[i],
                inverter_maxpower[i], dt, batteries[i]
            )
            dis_own_W[i] = dch_w

        LL = np.maximum(residual - dis_own_W, 0.0)

        # Step 3: PV → community loads (previously step 4)
        pv_rest = np.zeros(N) #
        surplus = p - direct #
        SE = surplus.copy()
        LL_tot = LL.sum()
        SE_tot = SE.sum()
        comm_dir_pv = np.zeros(N)
        comm_dir_load = np.zeros(N)

        if (SE_tot > 0 and LL_tot > 0):
            comm_dir_pv = proportional_allocate(LL_tot, SE.copy(), SE.copy())
            comm_dir_load = proportional_allocate(comm_dir_pv.sum(), LL.copy(), LL.copy())

            LL -= comm_dir_load
        pv_rest = surplus - comm_dir_pv  #
        #print('PV rest:', surplus)
        served_comm[t, :] += comm_dir_load
        comm_pv_to_load_to_self[t, :] += comm_dir_load
        comm_pv_to_load_to_others[t, :] += comm_dir_pv

        # Step 4: Community battery discharge sharing (previously step 6)
        dis_comm_bat_W = np.zeros(N)
        LL_comm = LL.copy()
        LL_total = LL_comm.sum()
        served_dist = np.zeros(N)

        if LL_total > 1e-12:
            avail_dis = np.minimum.reduce([
                battery_maxpower - dis_own_W,
                inverter_maxpower,
                np.maximum(soc_mid / (dt) * battery_eff, 0.0)
            ])

            P_from_bess = min(LL_total, avail_dis.sum())
            if P_from_bess > 1e-12:
                dis_comm = proportional_allocate(P_from_bess, avail_dis, avail_dis)
                #dis_comm = np.minimum.reduce([dis_comm, avail_dis])  # Make sure battery constraints are followed 
                LL_frac = LL_comm / LL_total
                served_dist = dis_comm.sum() * LL_frac
                soc_mid -= dis_comm * dt / battery_eff
                #print(LL_frac, dis_comm, LL)
                LL -= served_dist
 
                battery_capacities = np.array([batteries[i].C for i in range(N)])
                soc_mid = np.clip(soc_mid, 0, battery_capacities)
                served_comm[t, :] += served_dist
                comm_bat_to_load_to_self[t, :] += served_dist

                for i in range(N):
                    dis_comm_bat_W[i] = dis_comm[i]
                comm_bat_to_load_to_others[t, :] = dis_comm
                #print('****')

        # Step 5: Own PV → own battery (previously step 3)
        char_own_bat_W = np.zeros(N)
        soc_after = soc_mid.copy()
        charge_caps = np.zeros(N)
        pv_rest_own_charge = pv_rest.copy()

        for i in range(N):
            # Check that the battery is not discharging
            if dis_own_W[i] + dis_comm_bat_W[i] <= 0:
                cap_headroom_W = max((batteries[i].C - soc_mid[i]) / (dt * battery_eff[i]), 0.0)
                charge_caps[i] = min(battery_maxpower[i], inverter_maxpower[i], cap_headroom_W)
                ch_w = min(pv_rest_own_charge[i], charge_caps[i])
                soc_after[i] += ch_w * dt * battery_eff[i]
                #pv_rest[i] = surplus[i] - ch_w
                pv_rest_own_charge[i] = pv_rest[i] - ch_w #
                char_own_bat_W[i] = ch_w

        # Step 6: PV → community batteries (previously step 5)
        char_comm_bat_W = np.zeros(N)
        #SE2 = SE_tot - comm_dir.sum()
        SE2 = pv_rest_own_charge.sum()
        if SE2 > 0:
            ch_caps = np.zeros(N)
            for i in range(N):
                # Check that the battery is not discharging
                if dis_own_W[i] + dis_comm_bat_W[i] <= 0: 
                    cap_headroom_W = max(
                        (batteries[i].C - soc_after[i]) / (dt * battery_eff[i]), 0.0
                    )
                    ch_caps[i] = min(battery_maxpower[i] - char_own_bat_W[i], inverter_maxpower[i], cap_headroom_W)

            alloc = proportional_allocate(SE2, ch_caps, ch_caps)
            pv_to_comm_batteries = proportional_allocate(alloc.copy().sum(), pv_rest_own_charge.copy(), pv_rest_own_charge.copy())
            for i in range(N):
                # Check that the battery is not discharging
                if dis_own_W[i] + dis_comm_bat_W[i] <= 0:
                    soc_after[i] = min(
                        batteries[i].C,
                        soc_after[i] + alloc[i] * dt * battery_eff[i]
                    )
                    char_comm_bat_W[i] = alloc[i]
            comm_pv_to_bat_to_self[t, :] = alloc
            comm_pv_to_bat_to_others[t, :] = pv_to_comm_batteries

        # Update batteries
        for i in range(N):
            #print('Ch own bat | ', 'Ch comm bat | ', 'Dch own bat | ', 'Dch comm bat')
            #print(char_own_bat_W[i], char_comm_bat_W[i], dis_own_W[i], dis_comm_bat_W[i])
            batteries[i].update(charge_power=(char_own_bat_W[i] + char_comm_bat_W[i]), discharge_power=(dis_own_W[i] + dis_comm_bat_W[i]), 
                                dt_hours=dt, temp=indoor_temperature[t, i]+273.15)
            self_discharges[t, i] = batteries[i].self_discharge_history[t]
            soc_battery_class[t, i] = batteries[i].ch
            soc_after_before = soc_after[i]
            soc_after[i] = min(batteries[i].C, soc_after[i])
            capacity[t, i] = batteries[i].C
            disappeared[t, i] = soc_after_before - soc_after[i]

        lost[t, :] = LL
        served_self[t, :] = direct + dis_own_W
        soc[t, :] = soc_after

    return {
            "soc_Wh": soc,
            "lost_W": lost,
            "served_W_self": served_self,
            "served_W_comm": served_comm,
            "comm_pv_to_load_to_others": comm_pv_to_load_to_others,
            "comm_pv_to_load_to_self": comm_pv_to_load_to_self,
            "comm_pv_to_bat_to_others": comm_pv_to_bat_to_others,
            "comm_pv_to_bat_to_self": comm_pv_to_bat_to_self,
            "comm_bat_to_load_to_others": comm_bat_to_load_to_others,
            "comm_bat_to_load_to_self": comm_bat_to_load_to_self,
            "disappeared": disappeared,
            "soc_Wh_battery_class": soc_battery_class,
            "capacity": capacity,
            "load_W": dem_W,
            "production": pv_W,
            "indoor_temperature": indoor_temperature,
            "self_dishcarge": self_discharges
            #"idx": idx
        }


# ============================================================
# MAIN
# ============================================================

def main():

    degradations = [True]
    cabins_list = [4]
    degradation_models = ['Naumann']
    soc_idle_limits = [0.8, 0.5, 0.2]

    # ---- Load input files ----
    # These were used for cabin paper
    dem = pd.read_excel(os.path.join('Data', "Demand_cottages_20years.xlsx"), engine="openpyxl", header=None)
    pv_1kWp  = pd.read_excel(os.path.join('Data', "PV_gala_south_mean_20years.xlsx"), engine="openpyxl", header=None)
    tech = pd.read_excel(os.path.join('Data', "Technical_data_4_double_battery_size.xlsx"), engine="openpyxl", header=0)
    air_temp = pd.read_excel(os.path.join('Data', 'Temperatures_20years.xlsx'), engine="openpyxl")
    dem = dem.iloc[1:].reset_index(drop=True).astype(float)
    pv_1kWp  = pv_1kWp.iloc[1:].reset_index(drop=True).astype(float)
    indoor_temperature = get_indoor_temperature(air_temp.copy(), dem.copy())

    T, N = dem.shape
    cols = tech.columns[1:1+N]
    param_col = tech.columns[0]

    def get_param(name):
        row = tech[tech[param_col].astype(str).str.strip() == name]
        return row[cols].astype(float).values[0]

    battery_capacity   = get_param("Battery capacity in Wh")
    battery_startSOC   = get_param("Battery start value in percent") / 100.0
    battery_maxpower   = get_param("Battery max power in W")
    battery_eff        = np.array([Battery(True, 'Naumann', 10, False, 10, 10).eff(1)]*N)  # Get the efficincies from the battery model
    inverter_maxpower  = get_param("Inverter max power in W")
    PV_capacity        = get_param("PV capacity in Wp")
    
    pv = pd.concat(
        [pv_1kWp * c for c in PV_capacity],
        axis=1
    )
    pv.columns = dem.columns

    for degradation in degradations:
        for cabins in cabins_list:
            for degradation_model in degradation_models:
                for soc_idle_limit in soc_idle_limits:
                    print(dem.shape, pv.shape, len(battery_capacity), len(battery_startSOC),
                        len(battery_eff), len(inverter_maxpower))
                    dem = dem.iloc[:,:cabins]
                    pv = pv.iloc[:,:cabins]
                    battery_capacity = battery_capacity[:cabins]
                    battery_startSOC = battery_startSOC[:cabins]
                    battery_maxpower = battery_maxpower[:cabins]
                    battery_eff = battery_eff[:cabins]
                    inverter_maxpower = inverter_maxpower[:cabins]

                    degr_str = ''
                    if degradation:
                        degr_str = f'_{degradation_model}'

                    # ---- Run the two models ----
                    #res_off  = simulate_offgrid(dem.values, pv.values, indoor_temperature,
                    #                            battery_capacity, battery_startSOC,
                    #                            battery_maxpower, battery_eff, 
                    #                            inverter_maxpower, degradation_model,
                    #                            include_degradation=degradation,
                    #                            idle_soc_limit=soc_idle_limit)

                    ## Write to binary file
                    #with open(os.path.join("Results", "off_grid", f"res_off{degr_str}_{cabins}_{int(soc_idle_limit*100)}.pkl"), "wb") as file:
                    #    pickle.dump(res_off, file)
                    #    print(file)

                res_rule = simulate_rulebased(dem.values, pv.values, indoor_temperature,
                                            battery_capacity, battery_startSOC,
                                            battery_maxpower, battery_eff, 
                                            inverter_maxpower, degradation_model,
                                            include_degradation=degradation)


                with open(os.path.join("Results", "rule_based", f"res_rule{degr_str}_{cabins}.pkl"), "wb") as file:
                    pickle.dump(res_rule, file)
                    print(file)


# Run main
if __name__ == "__main__":
    # Sanity banner to confirm you’re running the latest file
    print(">>> RUNNING CLEANED SCRIPT —", __file__)
    main()
 