import logging
from typing import List, Tuple
from .vm import VM, VMState
from .scaling import ScalingStrategy

class LoadBalancer:
    """
    Dispatches requests to VMs and applies a scaling strategy.
    """
    def __init__(self, vms: List[VM], scaling_strategy: ScalingStrategy = None, 
                 max_vms: int = 10, max_starting_vms: int = 2) -> None:
        """
        Initialize a new load balancer.
        """
        self.vms = vms
        self.scaling_strategy = scaling_strategy
        self.next_vm_id = max([vm.id for vm in vms], default=-1) + 1
        self.vm_max_cpu = 100  # Default CPU capacity
        self.vm_max_ram = 200  # Default RAM capacity
        self.max_vms = max_vms  # Maximum number of VMs allowed
        self.max_starting_vms = max_starting_vms  # Maximum number of VMs in STARTING state

    def dispatch(self, req) -> None:
        """
        Dispatch the request to the VM with the smallest queue.
        Prioritize RUNNING VMs, fall back to STARTING VMs if needed.
        """
        running_vms = [vm for vm in self.vms if vm.can_process_requests()]
        if not running_vms:
            # If no running VMs, add to any VM that's starting
            starting_vms = [vm for vm in self.vms if vm.state == VMState.STARTING]
            if starting_vms:
                target_vm = min(starting_vms, key=lambda vm: len(vm.queue))
                target_vm.add_request(req)
            else:
                logging.warning("No available VMs to handle request!")
                return
        else:
            target_vm = min(running_vms, key=lambda vm: len(vm.queue))
            target_vm.add_request(req)

    def update_vm_states(self) -> None:
        """
        Update states of all VMs and remove terminated ones.
        """
        for vm in self.vms:
            vm.update_state()
        # Remove any terminated VMs
        self.vms = [vm for vm in self.vms if vm.state != VMState.TERMINATED]

    def get_vms_by_state(self) -> Tuple[int, int, int]:
        """
        Get counts of VMs in each state.
        """
        starting = len([vm for vm in self.vms if vm.state == VMState.STARTING])
        running = len([vm for vm in self.vms if vm.state == VMState.RUNNING])
        terminated = len([vm for vm in self.vms if vm.state == VMState.TERMINATED])
        return starting, running, terminated

    def scale_cluster(self, cycle: int) -> None:
        """
        Apply the scaling strategy if set.
        """
        if not self.scaling_strategy:
            logging.warning("No scaling strategy set!")
            return

        # Update VM states before scaling decisions
        self.update_vm_states()

        # Get current VM counts by state
        starting_vms, running_vms, _ = self.get_vms_by_state()
        logging.info(f"[Cycle {cycle}] Current VMs - Starting: {starting_vms}, Running: {running_vms}")
        
        # Get scaling decisions
        vms_to_add, vms_to_remove = self.scaling_strategy.scale(self.vms, cycle)

        # Limit new VMs based on both max total VMs and max starting VMs
        current_vm_count = len(self.vms)
        allowed_new_vms = min(
            self.max_vms - current_vm_count,  # Limit by max total VMs
            self.max_starting_vms - starting_vms  # Limit by max starting VMs
        )
        vms_to_add = max(0, min(vms_to_add, allowed_new_vms))

        # Only add new VMs if we're under the starting VM limit
        if vms_to_add > 0:
            logging.info(f"[Cycle {cycle}] Current VMs - Total: {current_vm_count}, Starting: {starting_vms}, Running: {running_vms}")

        # Remove VMs (ensuring at least one running VM remains)
        running_vms = [vm for vm in self.vms if vm.can_process_requests()]
        for vm in vms_to_remove:
            if len(running_vms) > 1:
                vm.terminate()
                running_vms.remove(vm)  # Remove from running list to maintain count
                logging.info(f"[Cycle {cycle}] Scale DOWN: VM {vm.id} marked for termination")

        # Add new VMs in STARTING state
        for _ in range(vms_to_add):
            new_vm = VM(vm_id=self.next_vm_id, max_cpu=self.vm_max_cpu, max_ram=self.vm_max_ram)
            self.vms.append(new_vm)
            logging.info(f"[Cycle {cycle}] Scale UP: Added VM {new_vm.id} (starting)")
            self.next_vm_id += 1