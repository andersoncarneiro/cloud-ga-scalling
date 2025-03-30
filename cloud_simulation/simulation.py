import logging
import numpy as np
from typing import List, Optional
from pydantic import BaseModel

from cloud_simulation.service import Service, ServiceRequest
from cloud_simulation.load_balancer import LoadBalancer
from cloud_simulation.vm import VM
from cloud_simulation.scaling import ScalingStrategy
from cloud_simulation.traffic import TrafficPattern, DailyPattern

logging.basicConfig(level=logging.INFO)

class SimulationResults(BaseModel):
    """
    Standard format for simulation results to ensure consistency.
    """
    vm_history: List[int]
    response_history: List[float]
    requests_history: List[int]
    error_budget_history: List[float]
    queue_length_history: List[float]
    
    class Config:
        arbitrary_types_allowed = True

class Simulation:
    """
    Runs a cloud simulation over a number of cycles with specific parameters.
    """
    def __init__(self, 
                 total_cycles: int, 
                 lb: LoadBalancer, 
                 service: Service, 
                 traffic_pattern: Optional[TrafficPattern] = None) -> None:
        """
        Initialize a new simulation.
        
        Args:
            total_cycles: Total number of cycles to simulate
            lb: LoadBalancer instance to distribute requests
            service: Service being simulated
            traffic_pattern: Traffic pattern to use (defaults to DailyPattern)
        """
        self.total_cycles = total_cycles
        self.lb = lb
        self.service = service
        # Use default daily pattern if none is provided
        self.traffic_pattern = traffic_pattern or DailyPattern()
        self.requests_history = []

    def dispatch_requests(self, cycle: int) -> None:
        """
        Generate and dispatch requests for the current cycle.

        """
        # Get expected request rate for this cycle
        lam = self.traffic_pattern.get_mean_arrivals(cycle, self.total_cycles)
        # Generate random number of arrivals based on Poisson distribution
        arrivals = np.random.poisson(lam)
        self.requests_history.append(arrivals)
        
        # Create and dispatch each request
        for _ in range(arrivals):
            req = ServiceRequest(arrival_time=cycle, service=self.service)
            self.lb.dispatch(req)

    def process_cycle(self, cycle: int) -> None:
        """
        Process a single simulation cycle.

        """
        # Process requests in each VM and collect completed ones
        cycle_requests = []
        for vm in self.lb.vms:
            processed = vm.process_requests(cycle)
            cycle_requests.extend(processed)
        
        # Calculate aggregate metrics
        average_response_time = (sum(req.finish_time - req.arrival_time for req in cycle_requests) /
                          len(cycle_requests)) if cycle_requests else 0

        # Calculate resource utilization
        total_cpu = sum(vm.max_cpu for vm in self.lb.vms)
        total_ram = sum(vm.max_ram for vm in self.lb.vms)
        
        cpu = sum(vm.current_cpu_used for vm in self.lb.vms) / total_cpu if total_cpu > 0 else 0
        ram = sum(vm.current_ram_used for vm in self.lb.vms) / total_ram if total_ram > 0 else 0
        vms = len(self.lb.vms)
        queue_length = sum(len(vm.queue) for vm
                        in self.lb.vms)

        # Apply scaling decisions
        self.lb.scale_cluster(cycle)
        
        # Record metrics for this cycle
        self.service.record_metric(
            cycle=cycle, 
            average_response_time=average_response_time, 
            requests=len(cycle_requests), 
            cpu=cpu, 
            ram=ram, 
            vms=vms,
            queue_length=queue_length
        )

    def run(self) -> SimulationResults:
        """
        Run the simulation for the specified number of cycles.

        """
        for cycle in range(self.total_cycles):
            self.dispatch_requests(cycle)
            self.process_cycle(cycle)
            
        metrics = self.service.metrics.aggregated_metrics
        return SimulationResults(
            vm_history=metrics['vms'],
            response_history=metrics['response_times'],
            requests_history=metrics['requests'],
            error_budget_history=metrics['error_budgets'],
            queue_length_history=metrics['queue_length'],
            cpu_history=metrics['cpu'],
            ram_history=metrics['ram'],
        )


    @classmethod
    def run_simulation(cls, 
                      traffic_pattern: TrafficPattern, 
                      total_cycles: int = 300, 
                      verbose: bool = False, 
                      scaling_strategy: Optional[ScalingStrategy] = None, 
                      service: Optional[Service] = None,
                      initial_vms: int = 3,
                      vm_max_cpu: int = 100,
                      vm_max_ram: int = 200,
                      max_vms: int = 30,
                      max_starting_vms: int = 2) -> SimulationResults:
        """
        Factory method to create and run a simulation with the specified parameters.
        
        Args:
            traffic_pattern: Traffic pattern to use
            total_cycles: Total number of cycles to simulate
            verbose: Whether to print detailed results
            scaling_strategy: Strategy for scaling VMs
            service: Service being simulated
            initial_vms: Number of VMs to start with
            vm_max_cpu: Maximum CPU capacity per VM
            vm_max_ram: Maximum RAM capacity per VM
            max_vms: Maximum number of VMs allowed
            max_starting_vms: Maximum number of VMs that can be in STARTING state
            
        Returns:
            SimulationResults object with aggregated metrics
        """
        # Create initial VMs
        vms = [VM(vm_id=i, max_cpu=vm_max_cpu, max_ram=vm_max_ram) for i in range(initial_vms)]

        # Create load balancer
        lb = LoadBalancer(
            vms=vms, 
            scaling_strategy=scaling_strategy,
            max_vms=max_vms,
            max_starting_vms=max_starting_vms
        )
        
        # Set VM creation parameters on the load balancer
        lb.vm_max_cpu = vm_max_cpu
        lb.vm_max_ram = vm_max_ram
        
        # Create and run simulation
        simulation = cls(
            total_cycles=total_cycles, 
            lb=lb, 
            service=service, 
            traffic_pattern=traffic_pattern
        )

        return simulation.run()
