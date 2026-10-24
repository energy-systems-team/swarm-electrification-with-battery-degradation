import numpy as np

import sys
import rainflow
import math

from .Inverter import Inverter

# Import model strategies
from .Magnor import init_magnor, update_magnor
from .Naumann import init_naumann, update_naumann
from .Gasper import init_gasper, update_gasper

# MODEL = 'Magnor'  # 'Magnor' | 'Naumann' | 'Gasper'

class Battery:
    """Main battery class with shared state and utilities. Models are modular."""

    def __init__(self, include_degradation, model, capacity, rule_80_20, specific_price, battery_max_Crate):
        # Core config
        self.model = model

        # Battery constants
        self.include_degradation = include_degradation  # Boolean
        self.C_init = capacity  # battery initial capacity [kWh]
        self.C = capacity  # initialize battery capacity [kWh]
        self.MAX_CH_POWER = battery_max_Crate * capacity  # kW
        self.MAX_DCH_POWER = battery_max_Crate * capacity # kW
        self.CAPEX = specific_price * capacity # capital costs [EUR]
        self.END_OF_LIFE = 0.8  # Percentual capacity in the end of life [-]
        self.rule_80_20 = rule_80_20

        # Self-discharge parameters (T_k_ref*T_alpha**((T-Tref)/T_halving_diff))
        # self-discharge speed is doubled for every 10 degree increase
        self.T_alpha = 2
        self.T_halving_diff = 10
        self.T_k_ref = 2/100 # 2% per month

        # Physical constants accessible to model modules through self
        self.R = 8.314   # J/mol/K
        self.F = 96485   # C/mol
        self.Tref = 298.15  # K

        # Inverter
        self.inverter = Inverter(rated_power=self.MAX_CH_POWER)

        # Histories
        self.capacity_history = np.array([])
        self.aging_factor_history = np.array([])
        self.aging_factor_cyclic_history = np.array([])
        self.aging_factor_float_history = np.array([])
        self.deltaSOH_history = np.array([])
        self.deltaSOH_cyclic_history = np.array([])
        self.deltaSOH_float_history = np.array([])
        self.self_discharge_history = np.array([])

        # Cycle tracking
        self.operation_mode = 'idle'
        self.half_cycle_history = np.array([])  # keep SOC history
        self._activity_history = np.array([], dtype=bool)
        self.t_cyc = 0.0
        self.t_cyc_active = 0.0
        self._simulation_hour = -1

        # Energy state
        self.ch = 0.0
        self.previous_ch = 0.0

        # Model-specific initialization
        if self.model == 'Magnor':
            init_magnor(self)
        elif self.model == 'Naumann':
            init_naumann(self)
        elif self.model == 'Gasper':
            init_gasper(self)
        else:
            raise ValueError(f"Unknown model: {self.model}")

    # ---------------- Shared utilities ----------------
    def reset_history(self):
        self.capacity_history = np.array([])
        self.aging_factor_history = np.array([])
        self.aging_factor_cyclic_history = np.array([])
        self.aging_factor_float_history = np.array([])
        self.deltaSOH_history = np.array([])
        self.deltaSOH_cyclic_history = np.array([])
        self.deltaSOH_float_history = np.array([])
        self.half_cycle_history = np.array([])
        self._activity_history = np.array([], dtype=bool)
        self.self_discharge_history = np.array([])

    def eff(self, power: float) -> float:
        if power == 0:
            return 1.0
        inverter_eff = self.inverter.efficiency(power)
        return math.sqrt(0.9) * inverter_eff
    
    def __selfdischarge(self, temp, dt_hours: float = 1.0):
        self_discharge_rate = self.T_k_ref / ((24 / dt_hours) * 30)   # T_k_ref in unit of [-/month]
        min_C = 0.0
        if ((self.ch - min_C) - self_discharge_rate * self.C_init)<=0:
            self_discharge = 0
        else:
            self_discharge = self_discharge_rate * self.C_init
        self.ch = self.ch - self_discharge
        return self_discharge

    def __charge(self, power: float, dt_hours: float = 1.0):
        max_C = self.C
        self.ch = min(self.ch + power * self.eff(power) * dt_hours, max_C)

    def __discharge(self, power: float, dt_hours: float = 1.0):
        min_C = 0.0
        self.ch = max(self.ch - power / self.eff(power) * dt_hours, min_C)

    def cycle_check(self, charge_power: float, discharge_power: float, dt_hours: float = 1.0):
        prev_mode = self.operation_mode
        if charge_power > 0 and discharge_power <= 0:
            mode = 'charge'
        elif discharge_power > 0 and charge_power <= 0:
            mode = 'discharge'
        elif charge_power > 0 and discharge_power > 0:
            mode = 'charge' if charge_power >= discharge_power else 'discharge'
        else:
            mode = 'idle'
        if mode in ('charge', 'discharge'):
            self.t_cyc_active += dt_hours
        self.half_cycle_history = np.append(self.half_cycle_history, self.ch)
        self.t_cyc += dt_hours
        completed = ((prev_mode == 'charge' and mode == 'discharge') or
                     (prev_mode == 'discharge' and mode == 'charge'))
        result = {"completed": False, "DOC": 0.0, "hours": self.t_cyc, "active_hours": self.t_cyc_active}
        if completed and self.half_cycle_history.size >= 2:
            Csafe = self.C + 1e-10
            DOC = (self.half_cycle_history.max() - self.half_cycle_history.min()) / Csafe
            DOC = min(1.0, max(0.0, DOC))
            hours = max(self.t_cyc, 1e-9)
            self.half_cycle_history = np.array([self.ch])
            self.t_cyc = 0.0
            result = {"completed": True, "DOC": DOC, "hours": hours, "active_hours": self.t_cyc_active}
        self.operation_mode = mode
        return result

    def cycle_check_rainflow_old(self, dt_hours: float = 1.0, active_now: bool = False):
        self.half_cycle_history = np.append(self.half_cycle_history, self.ch / (self.C + 1e-10))
        self._activity_history = np.append(self._activity_history, bool(active_now))
        cycles = []
        closed = []
        for rng, mean, count, start, end in rainflow.extract_cycles(self.half_cycle_history):
            i0 = int(start)
            i1 = int(end)
            if i1 <= i0:
                continue
            DOC = min(1.0, max(0.0, rng))
            hours = (i1 - i0) * dt_hours
            active_steps = int(self._activity_history[i0:i1].sum())
            active_hours = max(active_steps, 0) * dt_hours
            cycles.append({
                "DOC": DOC,
                "hours": max(hours, 1e-12),
                "active_hours": max(active_hours, 1e-12),
                "count": count,
                "soc_mean": float(mean)
            })
            closed.append(end)
        if closed:
            last_closed = int(max(closed))
            self.half_cycle_history = self.half_cycle_history[last_closed:]
            self._activity_history = self._activity_history[last_closed:]
        return cycles

    def cycle_check_rainflow(self, dt_hours: float = 1.0, active_now: bool = False):
        self.half_cycle_history = np.append(self.half_cycle_history, self.ch / (self.C + 1e-10))
        self._activity_history = np.append(self._activity_history, bool(active_now))
        self._simulation_hour += 1

        # Perform rainflow every 30th day (~once a month) 
        if self._simulation_hour % (30*24) != 0:
            return None
        
        cycles = []
        closed = []
        for rng, mean, count, start, end in rainflow.extract_cycles(self.half_cycle_history):
            i0 = int(start)
            i1 = int(end)
            if i1 <= i0:
                continue
            DOC = min(1.0, max(0.0, rng))
            hours = (i1 - i0) * dt_hours
            active_steps = int(self._activity_history[i0:i1].sum())
            active_hours = max(active_steps, 0) * dt_hours
            cycles.append({
                "DOC": DOC,
                "hours": max(hours, 1e-12),
                "active_hours": max(active_hours, 1e-12),
                "count": count,
                "soc_mean": float(mean)
            })
            closed.append(end)
        if closed:
            last_closed = int(max(closed))
            self.half_cycle_history = self.half_cycle_history[self._simulation_hour:]
            self._activity_history = self._activity_history[self._simulation_hour:]
        return cycles


    # ---------------- Unified update entrypoint ----------------
    def update(self, charge_power: float, discharge_power: float, include_float_degradation: bool = True, dt_hours: float = 1.0, temp: float = 298.15):
        if charge_power * discharge_power > 1e-5:
            sys.exit(f"ERROR: charge and discharge occurring simultaneously: {charge_power} vs {discharge_power}")
        self.previous_ch = self.ch
        self.capacity_history = np.append(self.capacity_history, self.C)
        
        self.__charge(charge_power, dt_hours=dt_hours)
        self.__discharge(discharge_power, dt_hours=dt_hours)
        
        self_dishcarge = self.__selfdischarge(dt_hours=dt_hours, temp=temp)
        self.self_discharge_history = np.append(self.self_discharge_history, self_dishcarge)

        if self.model == 'Magnor':
            update_magnor(self, charge_power, discharge_power, include_float_degradation, dt_hours, temp)
        elif self.model == 'Naumann':
            update_naumann(self, charge_power, discharge_power, include_float_degradation, dt_hours, temp)
        elif self.model == 'Gasper':
            update_gasper(self, charge_power, discharge_power, include_float_degradation, dt_hours, temp)
        else:
            raise ValueError(f"Unknown model: {self.model}")

