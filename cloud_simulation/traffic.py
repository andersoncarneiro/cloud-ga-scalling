from abc import ABC, abstractmethod

class TrafficPattern(ABC):
    """Abstract base class for traffic generation patterns."""
    
    @abstractmethod
    def get_mean_arrivals(self, cycle: int, total_cycles: int) -> float:
        """
        Returns the expected number of arrivals per cycle based on the pattern.
        
        Args:
            cycle: Current simulation cycle
            total_cycles: Total number of simulation cycles
            
        Returns:
            Mean number of arrivals for this cycle
        """
        pass
    
    def __str__(self) -> str:
        """Return the class name as a string representation"""
        return self.__class__.__name__


class SteadyTraffic(TrafficPattern):
    """Traffic pattern with consistent arrival rate."""
    
    def __init__(self, mean_arrivals: float):
        """
        Args:
            mean_arrivals: Constant mean arrival rate
        """
        self.mean_arrivals = mean_arrivals
        
    def get_mean_arrivals(self, cycle: int, total_cycles: int) -> float:
        return self.mean_arrivals


class DailyPattern(TrafficPattern):
    """Traffic pattern that follows daily peaks and valleys with optional spikes."""
    
    def __init__(self, peak_arrivals: float = 12.0, off_peak_arrivals: float = 3.0,
                 peak_start_hour: int = 8, peak_end_hour: int = 20, 
                 spike_arrivals: float = None, spike_hours: list = None):
        """
        Args:
            peak_arrivals: Mean arrivals during peak hours
            off_peak_arrivals: Mean arrivals during off-peak hours
            peak_start_hour: Hour when peak period starts (0-23)
            peak_end_hour: Hour when peak period ends (0-23)
            spike_arrivals: Mean arrivals during spike hours (if None, no spikes)
            spike_hours: List of hours when spikes occur (0-23), e.g. [9, 12, 17]
        """
        self.peak_arrivals = peak_arrivals
        self.off_peak_arrivals = off_peak_arrivals
        self.peak_start_hour = peak_start_hour
        self.peak_end_hour = peak_end_hour
        self.spike_arrivals = spike_arrivals
        self.spike_hours = spike_hours or []
        
    def get_mean_arrivals(self, cycle: int, total_cycles: int) -> float:
        hour = int((cycle / total_cycles) * 24)
        
        # Check if current hour is a spike hour
        if self.spike_arrivals is not None and hour in self.spike_hours:
            return self.spike_arrivals
            
        # Check if current hour is in peak period
        if self.peak_start_hour <= hour < self.peak_end_hour:
            return self.peak_arrivals
        
        # Otherwise return off-peak arrival rate
        return self.off_peak_arrivals
        
    def __str__(self):
        base_info = f"DailyPattern(peak={self.peak_arrivals}, off_peak={self.off_peak_arrivals}, "
        base_info += f"peak_hours={self.peak_start_hour}-{self.peak_end_hour}"
        
        if self.spike_arrivals is not None:
            base_info += f", spike={self.spike_arrivals}, spike_hours={self.spike_hours}"
            
        return base_info + ")"



class GradualGrowth(TrafficPattern):
    """Traffic pattern with gradual growth over time."""
    
    def __init__(self, initial_arrivals: float = 3.0, final_arrivals: float = 20.0, 
                 growth_type: str = "linear"):
        """
        Args:
            initial_arrivals: Mean arrival rate at the start
            final_arrivals: Mean arrival rate at the end
            growth_type: Type of growth - "linear" or "exponential"
        """
        self.initial_arrivals = initial_arrivals
        self.final_arrivals = final_arrivals
        self.growth_type = growth_type
        
    def get_mean_arrivals(self, cycle: int, total_cycles: int) -> float:
        progress = cycle / total_cycles
        
        if self.growth_type == "exponential":
            # Exponential growth: initial * (final/initial)^progress
            return self.initial_arrivals * (self.final_arrivals/self.initial_arrivals) ** progress
        else:
            # Default: linear growth
            return self.initial_arrivals + (self.final_arrivals - self.initial_arrivals) * progress

