from typing import List, Optional
from .service import ServiceRequest
from enum import Enum

class VMState(Enum):
    """
    Represents the various states a VM can be in.
    """
    STARTING = "starting"
    RUNNING = "running"
    TERMINATED = "terminated"

class VM:
    """
    Represents a virtual machine with separate CPU and RAM capacities.
    """
    def __init__(self, vm_id: int, max_cpu: float, max_ram: float) -> None:
        self.id = vm_id
        self.max_cpu = max_cpu
        self.max_ram = max_ram
        self.queue: List[ServiceRequest] = []
        self.idle_cycles: int = 0
        self.current_request: Optional[ServiceRequest] = None
        self.current_cpu_used: float = 0.0
        self.current_ram_used: float = 0.0
        self.state = VMState.STARTING
        self.cycles_in_current_state = 0

    def update_state(self) -> None:
        """Update VM state based on cycles spent in current state"""
        self.cycles_in_current_state += 1
        if self.state == VMState.STARTING and self.cycles_in_current_state >= 10:
            self.state = VMState.RUNNING
            self.cycles_in_current_state = 0

    def can_process_requests(self) -> bool:
        """Check if VM can process requests"""
        return self.state == VMState.RUNNING

    def terminate(self) -> None:
        """Mark VM for termination"""
        self.state = VMState.TERMINATED
        self.queue = []  # Clear any remaining requests

    def add_request(self, req: ServiceRequest) -> None:
        """Add a request to the VM's queue."""
        self.queue.append(req)

    def finish_current_request(self, current_cycle: int) -> List[ServiceRequest]:
        """Finish the current request if its processing time is over."""
        processed = []
        if self.current_request and current_cycle >= self.current_request.start_time + self.current_request.processing_time:
            self.current_request.finish_time = current_cycle
            processed.append(self.current_request)
            self.current_request = None
            self.current_cpu_used = 0.0
            self.current_ram_used = 0.0
        return processed

    def try_start_next_request(self, current_cycle: int) -> None:
        """Attempt to start processing the next request in the queue."""
        if not self.current_request and self.queue and self.can_process_requests():
            next_req = self.queue[0]
            if next_req.cpu_consumption <= self.max_cpu and next_req.ram_consumption <= self.max_ram:
                self.current_request = self.queue.pop(0)
                self.current_request.start_time = current_cycle
                self.current_cpu_used = self.current_request.cpu_consumption
                self.current_ram_used = self.current_request.ram_consumption

    def update_idle_cycles(self) -> None:
        """Increment idle cycles if no request is processing and the queue is empty."""
        if not self.current_request and not self.queue:
            self.idle_cycles += 1
        else:
            self.idle_cycles = 0

    def process_requests(self, current_cycle: int) -> List[ServiceRequest]:
        """
        Process requests for the current cycle.
        Returns a list of completed requests in this cycle.
        """
        if not self.can_process_requests():
            # Update state, but don't process anything if not in RUNNING state
            self.update_state()
            return []
            
        # First check if current request is finished
        processed_requests = self.finish_current_request(current_cycle)
        
        # Try to start next request if possible
        self.try_start_next_request(current_cycle)
        
        # Update idle counter
        self.update_idle_cycles()
        
        return processed_requests
