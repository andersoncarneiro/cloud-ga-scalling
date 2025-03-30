import unittest
import random
from cloud_simulation.service import SLO, Service, RequestConsumption, ServiceRequest
from cloud_simulation.vm import VM, VMState
from cloud_simulation.load_balancer import LoadBalancer
from cloud_simulation.scaling import ScalingPredicate,  WeightedPredicateBasedScaling


class TestCloudSimulation(unittest.TestCase):
    """Test cloud simulation components."""
    
    def setUp(self):
        """Set up common objects."""
        # Create a service with SLO
        self.slo = SLO(objective=0.95, threshold=10, total=100)
        self.request_consumption = RequestConsumption(cpu=70, ram=30)
        self.service = Service(
            name="TestService", 
            slo=self.slo, 
            request_consumption=self.request_consumption
        )
        
        # Create VMs
        self.vm = VM(vm_id=1, max_cpu=100, max_ram=200)
        # Ensure the VM is in RUNNING state
        self.vm.state = VMState.RUNNING
        
    def test_service_creates_metrics(self):
        """Test if the service correctly creates and stores metrics."""
        # Record a metric
        self.service.record_metric(
            cycle=1, 
            average_response_time=5.0, 
            requests=10, 
            cpu=0.6, 
            ram=0.4, 
            vms=3,
            queue_length=2
        )
        
        # Check if the metric was stored
        self.assertIn(1, self.service.metrics.metrics)
        metric = self.service.metrics.metrics[1]
        
        # Verify metric values
        self.assertEqual(metric.average_response_time, 5.0)
        self.assertEqual(metric.requests, 10)
        self.assertEqual(metric.cpu, 0.6)
        self.assertEqual(metric.ram, 0.4)
        self.assertEqual(metric.vms, 3)
        self.assertEqual(metric.queue_length, 2)
        
        # Test aggregated metrics
        agg_metrics = self.service.metrics.aggregated_metrics
        self.assertEqual(agg_metrics["cycles"], [1])
        self.assertEqual(agg_metrics["response_times"], [5.0])
        self.assertEqual(agg_metrics["requests"], [10])


    def test_slo_initialization(self):
        """Test if SLO correctly calculates the initial error budget."""
        # Create SLO with 95% objective and 100 total cycles
        slo = SLO(objective=0.95, threshold=10, total=100)
        
        # Verify error budget is correctly calculated
        expected_error_budget = 100 * (1 - 0.95)  # 5
        self.assertEqual(slo.error_budget, expected_error_budget)

    def test_service_request_resource_consumption(self):
        """Test if service requests consume the correct amount of CPU and RAM."""
        # Fixed random seed for repeatability
        random.seed(42)
        request = ServiceRequest(arrival_time=1, service=self.service)
        
        # Verify resource consumption is within expected range (x*1.1)
        self.assertGreaterEqual(request.cpu_consumption, self.service.request_consumption.cpu)
        self.assertLessEqual(request.cpu_consumption, self.service.request_consumption.cpu * 1.1)
        
        self.assertGreaterEqual(request.ram_consumption, self.service.request_consumption.ram)
        self.assertLessEqual(request.ram_consumption, self.service.request_consumption.ram * 1.1)
        
        # Verify processing time calculation
        expected_processing_time = max(1, int((request.cpu_consumption + request.ram_consumption) / 100))
        self.assertEqual(request.processing_time, expected_processing_time)

    def test_error_budget_decreases_on_high_latency(self):
        """Test if error budget decreases when latency exceeds threshold."""
        # Record normal metric (below threshold)
        self.service.record_metric(
            cycle=1, 
            average_response_time=5.0,  # Below the threshold of 10
            requests=10, 
            cpu=0.6, 
            ram=0.4, 
            vms=3,
            queue_length=2
        )
        
        # Get initial error budget
        initial_error_budget = self.service.error_budget
        
        # Record high latency metric (above threshold)
        self.service.record_metric(
            cycle=2, 
            average_response_time=15.0,  # Above the threshold of 10
            requests=10, 
            cpu=0.7, 
            ram=0.5, 
            vms=3,
            queue_length=4
        )
        
        # Verify error budget decreased
        self.assertLess(self.service.error_budget, initial_error_budget)
        
        # Verify violation flag was set
        self.assertTrue(self.service.metrics.metrics[2].violation)

    def test_vm_scale_up_condition(self):
        """Test if a VM is added when a scale up condition is met."""
        # Create load balancer with a scaling strategy that always scales up
        vms = [VM(vm_id=i, max_cpu=100, max_ram=200) for i in range(3)]
        for vm in vms:
            vm.state = VMState.RUNNING
            
        # Create a scaling predicate that always evaluates to True
        def always_true(vms, cycle, service=None):
            return 100.0  # A high value that will always exceed threshold
            
        # Create a comparison function that always returns True
        def always_greater(x, t):
            return True
        always_greater.__name__ = "always_greater"
        
        # Create a scaling predicate
        scale_up_predicate = ScalingPredicate(always_true, 0.0, always_greater)
        
        # Create a scaling strategy that always scales up
        scaling_strategy = WeightedPredicateBasedScaling(
            scale_up_predicates_with_weights=[(scale_up_predicate, 1.0)],
            scale_down_predicates_with_weights=[],
            max_vms_to_add=2,  # Add up to 2 VMs
            max_vms_to_remove=0  # Don't remove any VMs
        )
        
        # Create load balancer with this strategy
        lb = LoadBalancer(vms=vms, scaling_strategy=scaling_strategy, max_vms=10)
        
        # Get initial VM count
        initial_vm_count = len(lb.vms)
        
        # Apply scaling
        lb.scale_cluster(cycle=1)
        
        # Verify VMs were added
        self.assertGreater(len(lb.vms), initial_vm_count)

    def test_vm_scale_down_condition(self):
        """Test if a VM is removed when a scale down condition is met."""
        # Create VMs with one being idle
        vms = [VM(vm_id=i, max_cpu=100, max_ram=200) for i in range(5)]
        for vm in vms:
            vm.state = VMState.RUNNING
        
        # Make some VMs idle
        vms[1].idle_cycles = 10
        vms[3].idle_cycles = 5
        
        # Create a scaling predicate that always evaluates to True for scale down
        def always_true(vms, cycle, service=None):
            return 0.0  # A low value that will always be below threshold
            
        # Create a comparison function that always returns True
        def always_less(x, t):
            return True
        always_less.__name__ = "always_less"
        
        # Create a scaling predicate
        scale_down_predicate = ScalingPredicate(always_true, 1.0, always_less)
        
        # Create a scaling strategy that always scales down
        scaling_strategy = WeightedPredicateBasedScaling(
            scale_up_predicates_with_weights=[],
            scale_down_predicates_with_weights=[(scale_down_predicate, 1.0)],
            max_vms_to_add=0,  # Don't add any VMs
            max_vms_to_remove=2  # Remove up to 2 VMs
        )
        
        # Create load balancer with this strategy
        lb = LoadBalancer(vms=vms, scaling_strategy=scaling_strategy, max_vms=10)
        
        # Get initial VM count
        initial_vm_count = len(lb.vms)
        
        # Apply scaling
        lb.scale_cluster(cycle=1)
        
        # Verify VMs were removed
        vm_count_after = len([vm for vm in lb.vms if vm.state != VMState.TERMINATED])
        self.assertLess(vm_count_after, initial_vm_count)

    def test_vm_state_transition(self):
        """Test if VM transitions from STARTING to RUNNING after required cycles."""
        # Create a new VM (starts in STARTING state)
        vm = VM(vm_id=10, max_cpu=100, max_ram=200)
        self.assertEqual(vm.state, VMState.STARTING)
        
        # Update state for 9 cycles (should remain in STARTING nas it takes 10 cycles to transition)
        for _ in range(9):
            vm.update_state()
        self.assertEqual(vm.state, VMState.STARTING)
        
        # Update state for 1 more cycle (should transition to RUNNING)
        vm.update_state()
        self.assertEqual(vm.state, VMState.RUNNING)

    def test_request_processing(self):
        """Test if a VM correctly processes requests and updates its state."""
        # Create a request
        request = ServiceRequest(arrival_time=1, service=self.service)
        
        # Add request to VM queue
        self.vm.add_request(request)
        self.assertEqual(len(self.vm.queue), 1)
        
        # Start processing request
        self.vm.try_start_next_request(current_cycle=2)
        self.assertIsNotNone(self.vm.current_request)
        self.assertEqual(self.vm.current_request, request)
        self.assertEqual(request.start_time, 2)
        
        # Verify CPU and RAM usage is updated
        self.assertEqual(self.vm.current_cpu_used, request.cpu_consumption)
        self.assertEqual(self.vm.current_ram_used, request.ram_consumption)
        
        # Finish processing (current_cycle should be >= start_time + processing_time)
        finish_cycle = 2 + request.processing_time
        processed = self.vm.finish_current_request(current_cycle=finish_cycle)
        
        # Verify request was processed
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0], request)
        self.assertEqual(request.finish_time, finish_cycle)
        
        # Verify VM resources were freed
        self.assertIsNone(self.vm.current_request)
        self.assertEqual(self.vm.current_cpu_used, 0.0)
        self.assertEqual(self.vm.current_ram_used, 0.0)

    def test_load_balancer_request_distribution(self):
        """Test if load balancer distributes requests to VMs with smallest queue."""
        # Create VMs with different queue lengths
        vms = [VM(vm_id=i, max_cpu=100, max_ram=200) for i in range(3)]
        for vm in vms:
            vm.state = VMState.RUNNING
        
        # Add some requests to VM queues
        for _ in range(3):
            vms[0].add_request(ServiceRequest(arrival_time=1, service=self.service))
        
        for _ in range(1):
            vms[1].add_request(ServiceRequest(arrival_time=1, service=self.service))
        
        # VM2 has an empty queue
        
        # Create load balancer with these VMs
        lb = LoadBalancer(vms=vms)
        
        # Create a new request
        request = ServiceRequest(arrival_time=1, service=self.service)
        
        # Dispatch request
        lb.dispatch(request)
        
        # Verify request went to VM with smallest queue (VM2)
        self.assertEqual(len(vms[2].queue), 1)
        self.assertEqual(vms[2].queue[0], request)




if __name__ == '__main__':
    unittest.main()