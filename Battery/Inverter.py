
class Inverter:
    """
    Not yet supported
    """

    def __init__(self, rated_power):
        # ********************************************************
        # Inverter parameters
        # ********************************************************
        self.P_RATED = rated_power  # Rated power [kW]

    
    def efficiency(self, input_power=None):
        """
        A function that gives the invertes efficiency with a given input power.
        """        
        return 0.97
    
    
    def convert(self, input_power):
        output_power =  input_power * self.efficiency(input_power)
        return output_power
    