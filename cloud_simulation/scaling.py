import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Optional
from .vm import VM, VMState
from .service import Service

class ScalingStrategy(ABC):
    """
    Abstract base class for all scaling strategies.
    Determines when to add or remove VMs based on system metrics.
    """
    @abstractmethod
    def scale(self, vms: List[VM], cycle: int) -> Tuple[int, List[VM]]:
        """
        Determines the number of VMs to add and returns a list of VMs to remove.
        
        Args:
            vms: Current list of virtual machines
            cycle: Current simulation cycle
            
        Returns:
            Tuple of (vms_to_add, list_of_vms_to_remove)
        """
        pass


class ScalingPredicate:
    """
    A predicate that evaluates a specific condition against a threshold.
    Used to determine scaling decisions based on system metrics.
    """
    def __init__(self, 
                 metric_fn: Callable[[List[VM], int, Optional[Service]], float], 
                 threshold: float, 
                 comparison_fn: Callable[[float, float], bool]) -> None:
        """
        Initialize a new scaling predicate.
        
        Args:
            metric_fn: Function that computes a metric from VMs, cycle, and service
            threshold: Value to compare the metric against
            comparison_fn: Function that compares metric value to threshold
        """
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.comparison_fn = comparison_fn
        
    def evaluate(self, vms: List[VM], cycle: int, service: Optional[Service] = None) -> bool:
        """
        Evaluate the predicate for the given VMs and cycle.
        
        Args:
            vms: Current list of virtual machines
            cycle: Current simulation cycle
            service: Optional service to provide context (e.g., for metrics)
            
        Returns:
            True if the predicate condition is satisfied, False otherwise
        """
        current_value = self.metric_fn(vms, cycle, service)
        
        logging.info(f"[Cycle {cycle}] Evaluating predicate - Current value: {current_value}, "
                    f"Threshold: {self.threshold}, comparison: {self.comparison_fn.__name__}")
        
        return self.comparison_fn(current_value, self.threshold)
    
    def __str__(self) -> str:
        """String representation of the predicate"""
        comparison_name = self.comparison_fn.__name__
        comparison_symbol = ">" if "gt" in comparison_name else "<"
        return f"Metric {comparison_symbol} {self.threshold}"


class WeightedPredicateBasedScaling(ScalingStrategy):
    """Scaling strategy that uses weighted predicates and a threshold to determine scaling actions."""
    
    def __init__(self, 
                scale_up_predicates_with_weights: List[Tuple[ScalingPredicate, float]],
                scale_down_predicates_with_weights: List[Tuple[ScalingPredicate, float]],
                service: Optional[Service] = None,
                max_vms_to_add: int = 2,
                max_vms_to_remove: int = 1,
                scale_up_threshold: float = 0.6,
                scale_down_threshold: float = 0.6):
        """
        Initialize weighted predicate scaling.
        """
        self.scale_up_predicates_with_weights = scale_up_predicates_with_weights
        self.scale_down_predicates_with_weights = scale_down_predicates_with_weights
        self.service = service
        self.max_vms_to_add = max_vms_to_add
        self.max_vms_to_remove = max_vms_to_remove
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold

    def evaluate_weighted_predicates(self, 
                                    predicates_with_weights: List[Tuple[ScalingPredicate, float]], 
                                    vms: List[VM], 
                                    cycle: int) -> float:
        """
        Evaluate predicates with weights to get an overall score.
        
        Returns:
            Score between 0 and 1 indicating how strongly the conditions favor scaling
        """
        if not predicates_with_weights:
            return 0.0
            
        total_weight = sum(weight for _, weight in predicates_with_weights)
        if total_weight == 0:
            return 0.0
            
        weighted_sum = 0
        for predicate, weight in predicates_with_weights:
            result = predicate.evaluate(vms, cycle, self.service)
            weighted_sum += weight * (1 if result else 0)
        
        return weighted_sum / total_weight

    def scale(self, vms: List[VM], cycle: int) -> Tuple[int, List[VM]]:
        """Determine scaling actions based on weighted predicate evaluation."""
        # Skip if no VMs to evaluate
        if not vms:
            return 0, []
            
        # Evaluate predicates and get scores
        scale_up_score = self.evaluate_weighted_predicates(
            self.scale_up_predicates_with_weights, vms, cycle)
        scale_down_score = self.evaluate_weighted_predicates(
            self.scale_down_predicates_with_weights, vms, cycle)
        
        # Log scores
        logging.info(f"[Cycle {cycle}] Weighted scaling scores - Up: {scale_up_score:.2f}, "
                    f"Down: {scale_down_score:.2f}")
        
        # Make scaling decisions
        should_scale_up = scale_up_score >= self.scale_up_threshold
        should_scale_down = scale_down_score >= self.scale_down_threshold
        
        # Standard scaling logic
        vms_to_add = self.max_vms_to_add if should_scale_up else 0
        vms_to_remove = []
        
        if should_scale_down:
            candidates = [vm for vm in vms 
                        if vm.state == VMState.RUNNING and vm.idle_cycles > 0 and len(vm.queue) == 0]
            candidates.sort(key=lambda vm: -vm.idle_cycles)
            vms_to_remove = candidates[:self.max_vms_to_remove]
            
        return vms_to_add, vms_to_remove
# --- Predicate Factory Functions ---

# These functions create specific predicates for scaling decisions.

def create_cpu_utilization_predicate(threshold: float, greater_than: bool = True) -> ScalingPredicate:
    """
    Create a predicate that evaluates CPU utilization against a threshold.
    
    Args:
        threshold: Percentage threshold (0-100)
        greater_than: If True, scales when utilization > threshold; 
                      if False, scales when utilization < threshold
                      
    Returns:
        ScalingPredicate instance
    """
    def get_cpu_utilization(vms: List[VM], cycle: int, service: Optional[Service] = None) -> float:
        if not vms:
            return 0
        total_cpu = sum(vm.current_cpu_used for vm in vms)
        max_cpu = sum(vm.max_cpu for vm in vms)
        return (total_cpu / max_cpu) * 100 if max_cpu > 0 else 0
        
    compare = (lambda x, t: x > t) if greater_than else (lambda x, t: x < t)
    compare.__name__ = "gt" if greater_than else "lt"
    return ScalingPredicate(get_cpu_utilization, threshold, compare)


def create_queue_length_predicate(threshold: float, greater_than: bool = True) -> ScalingPredicate:
    """
    Create a predicate that evaluates maximum queue length against a threshold.
    
    Args:
        threshold: Queue length threshold
        greater_than: If True, scales when queue length > threshold; 
                      if False, scales when queue length < threshold
                      
    Returns:
        ScalingPredicate instance
    """
    def get_max_queue_length(vms: List[VM], cycle: int, service: Optional[Service] = None) -> float:
        if not vms:
            return 0
        return max((len(vm.queue) for vm in vms), default=0)
        
    compare = (lambda x, t: x > t) if greater_than else (lambda x, t: x < t)
    compare.__name__ = "gt" if greater_than else "lt"
    return ScalingPredicate(get_max_queue_length, threshold, compare)


def create_error_budget_predicate(threshold: float, greater_than: bool = True) -> ScalingPredicate:
    """
    Create a predicate that evaluates service error budget against a threshold.
    
    Args:
        threshold: Error budget threshold
        greater_than: If True, scales when error budget > threshold; 
                      if False, scales when error budget < threshold
                      
    Returns:
        ScalingPredicate instance
    """
    def get_error_budget(vms: List[VM], cycle: int, service: Optional[Service] = None) -> float:
        if not service or not service.metrics.error_budget_history:
            return float('inf')
        return service.metrics.error_budget_history[-1]
        
    compare = (lambda x, t: x > t) if greater_than else (lambda x, t: x < t)
    compare.__name__ = "gt" if greater_than else "lt"
    return ScalingPredicate(get_error_budget, threshold, compare)


def create_response_time_predicate(threshold: float, greater_than: bool = True) -> ScalingPredicate:
    """
    Create a predicate that evaluates average response time against a threshold.
    
    Args:
        threshold: Response time threshold in cycles
        greater_than: If True, scales when response time > threshold; 
                      if False, scales when response time < threshold
                      
    Returns:
        ScalingPredicate instance
    """
    def get_avg_response_time(vms: List[VM], cycle: int, service: Optional[Service] = None) -> float:
        if not service or not service.metrics.aggregated_metrics['response_times']:
            return 0
        # Use a moving average of the last 5 cycles to smooth out spikes
        recent_times = service.metrics.aggregated_metrics['response_times'][-5:]
        return sum(recent_times) / len(recent_times) if recent_times else 0
    
    compare = (lambda x, t: x > t) if greater_than else (lambda x, t: x < t)
    compare.__name__ = "gt" if greater_than else "lt"
    return ScalingPredicate(get_avg_response_time, threshold, compare)

