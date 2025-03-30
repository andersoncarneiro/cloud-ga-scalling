import logging
from cloud_simulation.scaling import (
    WeightedPredicateBasedScaling,
    create_cpu_utilization_predicate,
    create_error_budget_predicate,
    create_queue_length_predicate,
    create_response_time_predicate

)


def decode_fixed_predicates(solution, genes_per_predicate, predicates_per_direction):
    """
    Decode a GA solution with fixed predicate slots for scale-up and scale-down decisions.
    
    Format:
    - First 12 genes: 4 scale-up predicates (3 genes each)
    - Last 12 genes: 4 scale-down predicates (3 genes each)
    
    Each predicate has:
    - Gene 0: pred_type ∈ {0-7} (combined metric and direction)
    - Gene 1: threshold ∈ {1-100}
    - Gene 2: weight ∈ {1-10}
    
    Predicate Types:
    0: CPU Utilization > threshold
    1: CPU Utilization < threshold
    2: Response Time > threshold
    3: Response Time < threshold
    4: Queue Length > threshold
    5: Queue Length < threshold
    6: Error Budget > threshold
    7: Error Budget < threshold
    
    Returns:
        Tuple of (scale_up_predicates_with_weights, scale_down_predicates_with_weights)
    """
    scale_up_predicates_with_weights = []
    scale_down_predicates_with_weights = []
    
 
    
    def create_predicate_with_weight(pred_type, threshold, weight, predicate_idx, is_scale_up):
        """Helper function to create a predicate with appropriate scaling and logging"""
        direction_str = "scale-up" if is_scale_up else "scale-down"
        
        # Ensure weight is positive and convert to float
        weight = max(1.0, float(weight))
        
        # Apply appropriate scaling to threshold based on predicate type
        if pred_type == 0 or pred_type == 1:  # CPU Utilization
            # Keep in range 0-100
            scaled_threshold = min(100, max(0, threshold))
            greater_than = (pred_type == 0)
            predicate = create_cpu_utilization_predicate(scaled_threshold, greater_than=greater_than)
            metric_name = "CPU Utilization"
            
        elif pred_type == 2 or pred_type == 3:  # Response Time
            # Response time should be scaled differently - typical range is 0-10
  
            scaled_threshold = threshold * 0.1
            greater_than = (pred_type == 2)
            predicate = create_response_time_predicate(scaled_threshold, greater_than=greater_than)
            metric_name = "Response Time"
            
        elif pred_type == 4 or pred_type == 5:  # Queue Length
            # Queue length typically doesn't need scaling
            scaled_threshold = threshold * 0.2
            greater_than = (pred_type == 4)
            predicate = create_queue_length_predicate(scaled_threshold, greater_than=greater_than)
            metric_name = "Queue Length"
            
        elif pred_type == 6 or pred_type == 7:  # Error Budget
            # Error budget typically ranges from 0-1, so scale accordingly
            scaled_threshold = threshold * 0.1  # Allow finer control over error budget thresholds
            greater_than = (pred_type == 6)
            predicate = create_error_budget_predicate(scaled_threshold, greater_than=greater_than)
            metric_name = "Error Budget"
            
        else:
            # Invalid type - default to CPU > threshold
            logging.error(f"Invalid predicate type: {pred_type}")
        
        # Log the created predicate for debugging
        compare_symbol = ">" if greater_than else "<"
        logging.debug(f"Created {direction_str} predicate {predicate_idx}: {metric_name} {compare_symbol} {scaled_threshold} (weight: {weight})")
        
        return predicate, weight
    
    # Process scale-up predicates (first half)
    for i in range(predicates_per_direction):
        start_idx = i * genes_per_predicate
        
        # Extract genes and ensure proper types
        pred_type = int(round(solution[start_idx]))
        threshold = float(solution[start_idx + 1])
        weight = float(solution[start_idx + 2])
        
        # Create and add predicate
        predicate, weight = create_predicate_with_weight(
            pred_type, threshold, weight, i, is_scale_up=True)
        scale_up_predicates_with_weights.append((predicate, weight))
    
    # Process scale-down predicates (second half)
    for i in range(predicates_per_direction):
        start_idx = (i + predicates_per_direction) * genes_per_predicate
        
        # Extract genes and ensure proper types
        pred_type = int(round(solution[start_idx]))
        threshold = float(solution[start_idx + 1])
        weight = float(solution[start_idx + 2])
        
        # Create and add predicate
        predicate, weight = create_predicate_with_weight(
            pred_type, threshold, weight, i, is_scale_up=False)
        scale_down_predicates_with_weights.append((predicate, weight))
    
    # Log summary of created predicates for better debugging
    logging.info(f"Created {len(scale_up_predicates_with_weights)} scale-up predicates and "
                f"{len(scale_down_predicates_with_weights)} scale-down predicates")
    
    return scale_up_predicates_with_weights, scale_down_predicates_with_weights

def create_weighted_predicate_scaling_strategy(
    solution, 
    genes_per_predicate,
    num_predicates,
    service=None,
    scale_up_threshold=0.7,
    scale_down_threshold=0.7,
    max_vms_to_add=2,
    max_vms_to_remove=1
) -> WeightedPredicateBasedScaling:
    """
    Creates a weighted predicate-based scaling strategy from a genetic algorithm solution.
        
    Returns:
        A configured WeightedPredicateBasedScaling strategy
    """
    # Use weighted predicates decoding to match the fitness function approach
    scale_up_predicates_with_weights, scale_down_predicates_with_weights = decode_fixed_predicates(
        solution, 
        genes_per_predicate=genes_per_predicate, 
        predicates_per_direction=int(num_predicates/2)
    )

    return WeightedPredicateBasedScaling(
        scale_up_predicates_with_weights=scale_up_predicates_with_weights,
        scale_down_predicates_with_weights=scale_down_predicates_with_weights,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        service=service,
        max_vms_to_add=max_vms_to_add,
        max_vms_to_remove=max_vms_to_remove
    )

# I am fixing the first predicate to scale up. At the half of the genome, I am fixing the first predicate to scale down.
# This is an attempt to make the GA more efficient by reducing the search space, with a starting point based on the baseline configuration
def generate_gene_space(baseline_latency_up_threshold, baseline_latency_down_threshold, num_predicates, genes_per_predicate, index_scale_down):
    gene_space = []
    variations = [0.9, 0.95, 1.0, 1.05, 1.1]

    scale_up_values = [baseline_latency_up_threshold * 10 * factor for factor in variations]
    scale_down_values = [baseline_latency_down_threshold * 10 * factor for factor in variations]
    for i in range(num_predicates * genes_per_predicate):
    # SCALE UP PREDICATES
        if i == 0:
            gene_space.append([2])  # Predicate type: Response Time > threshold (fixed)
        elif i == 1:    
            gene_space.append(scale_up_values)  # Threshold with small variations
        elif i == 2:
            gene_space.append(list(range(4, 6)))  # Allow 4 or 5 as weught to scale up
    
    # SCALE DOWN PREDICATES
        elif i == index_scale_down:
            gene_space.append([3])  # Predicate type: Response Time < threshold (fixed)
        elif i == index_scale_down + 1:
            gene_space.append(scale_down_values)  # Threshold with small variations
        elif i == index_scale_down + 2:
            gene_space.append(list(range(4, 6)))    # Allow 4 or 5 as weught to scale down
    
    # Other genes can vary freely
        else:
        # Define appropriate ranges for other genes based on their meaning
            if i % 3 == 0:  #
                gene_space.append(list(range(8)))  # All possible predicate types
            elif i % 3 == 1:  
                gene_space.append(list(range(1,101)))  # Threshold values
            else:  
                gene_space.append(list(range(1, 6)))
    return gene_space