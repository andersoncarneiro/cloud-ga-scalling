
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class SLO(BaseModel):
    """
    Service Level Objective model with error budget calculation.
    """
    type: str = "Latency"
    objective: float = Field(..., gt=0, le=1)
    threshold: float = Field(..., gt=0)
    total: int = Field(..., gt=0)
    error_budget: Optional[float] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        if self.error_budget is None:
            self.error_budget = self.total * (1 - self.objective)


class CycleMetric(BaseModel):
    """
    Metrics collected for each simulation cycle.
    """
    average_response_time: float
    error_budget: float
    violation: bool
    requests: int
    cpu: float
    ram: float
    vms: int
    queue_length: int


class ServiceMetrics:
    """Tracks and manages all metrics for a service."""
    def __init__(self) -> None:
        self.metrics = {} # Dict[int, CycleMetric] = {}

    @property
    def aggregated_metrics(self) -> dict:
        """
        Returns a dictionary of lists for each metric across all cycles.
        The keys are:
          - cycles: list of cycle numbers
          - response_times: list of average response times
          - error_budgets: list of error budget values
          - violations: list of booleans indicating violation occurrence
          - requests: list of request counts
          - cpu: list of CPU usage values
          - ram: list of RAM usage values
          - vms: list of VM counts
        """
        if not self.metrics:
            return {
                "cycles": [],
                "response_times": [],
                "error_budgets": [],
                "violations": [],
                "requests": [],
                "cpu": [],
                "ram": [],
                "vms": [],
                "queue_length": [],
            }
            
        cycles = sorted(self.metrics.keys())
        response_times = [self.metrics[cycle].average_response_time for cycle in cycles]
        error_budgets  = [self.metrics[cycle].error_budget for cycle in cycles]
        violations = [self.metrics[cycle].violation for cycle in cycles]
        requests = [self.metrics[cycle].requests for cycle in cycles]
        cpus = [self.metrics[cycle].cpu for cycle in cycles]
        rams = [self.metrics[cycle].ram for cycle in cycles]
        vms = [self.metrics[cycle].vms for cycle in cycles]
        queue_lengths = [self.metrics[cycle].queue_length for cycle in cycles]
        return {
            "cycles": cycles,
            "response_times": response_times,
            "error_budgets": error_budgets,
            "violations": violations,
            "requests": requests,
            "cpu": cpus,
            "ram": rams,
            "vms": vms,
            "queue_length": queue_lengths,
        }
    
    @property
    def error_budget_history(self) -> List[float]:
        return [self.metrics[cycle].error_budget for cycle in sorted(self.metrics.keys())] if self.metrics else []


class RequestConsumption(BaseModel):
    """
    Represents the CPU and RAM consumption of a request.
    """
    cpu: float = Field(..., gt=0)
    ram: float = Field(..., gt=0)


class Service:
    """Represents a service with SLO tracking capabilities."""
    def __init__(self, name: str, slo: SLO, request_consumption: RequestConsumption) -> None:
        self.name = name
        self.slo = slo
        self.latency_threshold = slo.threshold
        self.total_cycles = slo.total
        self.total_allowed_violations = int(self.total_cycles * (1 - self.slo.objective))
        self.metrics = ServiceMetrics()
        self.request_consumption = request_consumption

    @property
    def error_budget(self) -> float:
        return self.metrics.error_budget_history[-1] if self.metrics.error_budget_history else self.slo.error_budget
    
    def record_metric(self, cycle: int, average_response_time: float, requests: int, cpu: float, ram: float, vms: int, queue_length: int) -> None:
        violation = False
        error_budget = self.error_budget
        
        if average_response_time > self.latency_threshold:
            error_budget = error_budget -1
            violation = True

        metric = CycleMetric(
            average_response_time=average_response_time, 
            error_budget=error_budget, 
            violation=violation, 
            requests=requests, 
            cpu=cpu, 
            ram=ram, 
            vms=vms,
            queue_length=queue_length
        )
        self.metrics.metrics[cycle] = metric


class ServiceRequest:
    """
    Represents an incoming request belonging to a Service.
    Each request consumes CPU and RAM resources.
    """
    def __init__(self, arrival_time: int, service: Service) -> None:
        self.arrival_time = arrival_time  # Cycle when the request arrives.
        self.start_time = None            # Cycle when processing starts.
        self.finish_time = None           # Cycle when processing finishes.
        self.server_id = None             # ID of the VM that processes this request.
        self.service = service            # Service to which this request belongs.
        
        # Simulate resource consumption based on request definition.
        self.cpu_consumption = random.uniform(self.service.request_consumption.cpu, self.service.request_consumption.cpu*1.1)
        self.ram_consumption = random.uniform(self.service.request_consumption.ram, self.service.request_consumption.ram*1.1)
        
        # Calculate processing time based on resource requirements.
        self.processing_time = max(1, int((self.cpu_consumption + self.ram_consumption) / 100))

    def __str__(self) -> str:
        return (f"Request({self.service.name}, arrival={self.arrival_time}, "
                f"CPU={self.cpu_consumption:.2f}, RAM={self.ram_consumption:.2f})")


