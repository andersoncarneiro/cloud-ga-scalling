import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Dict, Optional
from cloud_simulation.simulation import SimulationResults


def plot_traffic_patterns(traffic_patterns: Dict, total_cycles: int = 300) -> None:
    """
    Generate visualization of different traffic patterns.
    """
    # Create cycle array
    cycles = np.arange(total_cycles)
    
    # Compute arrival rates for each pattern
    arrival_rates = {}
    for name, pattern in traffic_patterns.items():
        arrival_rates[name] = [pattern.get_mean_arrivals(cycle, total_cycles) for cycle in cycles]
    
    # Create comparison plot with all patterns
    plt.figure(figsize=(14, 6))
    for name, rates in arrival_rates.items():
        plt.plot(cycles, rates, label=name)
    
    plt.title("Comparison of Traffic Patterns")
    plt.xlabel("Cycle")
    plt.ylabel("Mean Arrivals")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_simulation_results(results: SimulationResults, title: Optional[str] = None) -> None:
    """
    Plot simulation results: VMs, response times, and error budget.
    
    """
    vm_history = results.vm_history
    response_history = results.response_history
    requests_history = results.requests_history
    error_budget_history = results.error_budget_history
    

    total_cycles = len(vm_history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Subplot 1: VMs, response time, and requests per cycle
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('Number of VMs', color='blue')
    l1 = ax1.plot(range(total_cycles), vm_history, 'b-', label='VM Count')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1_2 = ax1.twinx()
    ax1_2.set_ylabel('Avg Response Time (cycles)', color='red')
    l2 = ax1_2.plot(range(total_cycles), response_history, 'r-', label='Response Time')
    ax1_2.tick_params(axis='y', labelcolor='red')
    
    ax1_3 = ax1.twinx()
    ax1_3.spines['right'].set_position(('outward', 60))
    ax1_3.set_ylabel('Requests per Cycle', color='green')
    l3 = ax1_3.plot(range(total_cycles),
                    requests_history + [0]*(total_cycles - len(requests_history)),
                    'g-', label='Requests')
    ax1_3.tick_params(axis='y', labelcolor='green')
    
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.set_title(title or 'System Metrics Over Time')
    ax1.grid(True)
    
    # Subplot 2: Service Error Budget
    ax2.plot(range(total_cycles), error_budget_history, 'r-', label='Error Budget')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Remaining Error Budget')
    ax2.set_title('Error Budget Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_multi_run_comparison(results_list, baseline_results=None, metrics= ['response_time', 'vm_count', 'error_budget'], figsize=(14, 10)):
    """
    Create visualization for multiple GA runs compared to baseline.
    """

    
    # Restructure data for plotting
    results_by_pattern = {}
    
    # Transform the list of dictionaries to dictionary of lists
    for run_results in results_list:
        for pattern, result in run_results.items():
            if pattern not in results_by_pattern:
                results_by_pattern[pattern] = []
            results_by_pattern[pattern].append(result)
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]  
    
    # Metric extraction functions
    def get_metric_value(result, metric_name):
        if metric_name == 'response_time':
            # Calculate average response time from history
            return np.mean([r for r in result.response_history if r > 0])
        elif metric_name == 'vm_count':
            # Calculate average VM count
            return np.mean(result.vm_history)
        elif metric_name == 'error_budget':
            # Get final error budget
            return result.error_budget_history[-1]
        else:
            # Fallback to generic attribute
            return getattr(result, metric_name, 0)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Convert to DataFrame for plotting
        df_data = []
        for pattern, results in results_by_pattern.items():
            for result in results:
                # Extract the appropriate metric
                value = get_metric_value(result, metric)
                df_data.append({'Pattern': pattern, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # Create boxplot
        sns.boxplot(x='Pattern', y='Value', data=df, ax=ax)
        
        
        pattern_names = list(results_by_pattern.keys())
        baseline_values = []
        
        for pattern in pattern_names:
            if pattern in baseline_results:
                baseline_values.append(get_metric_value(baseline_results[pattern], metric))
            else:
                baseline_values.append(np.nan)
        
        # Plot baseline as red dashed line with diamonds
        valid_indices = [j for j, val in enumerate(baseline_values) if not np.isnan(val)]
        valid_values = [baseline_values[j] for j in valid_indices]
        
        if valid_values:
            ax.plot(valid_indices, valid_values, 'r--', label='Baseline')
            ax.scatter(valid_indices, valid_values, color='red', marker='D', s=50)
        
        # Add percentage differences
        for j, pattern in enumerate(pattern_names):
            if pattern in baseline_results and not np.isnan(baseline_values[j]):
                baseline = baseline_values[j]
                if baseline != 0:  # Avoid division by zero
                    # Calculate median of this metric for this pattern
                    median = np.median([get_metric_value(result, metric) 
                                        for result in results_by_pattern[pattern]])
                    
                    pct_diff = (median - baseline) / baseline * 100
                    
                    # Determine color based on improvement direction
                    if pct_diff < 0:
                        color = 'green'
                        label = f"{pct_diff:.1f}%"
                    else:
                        color = 'red'
                        label = f"+{pct_diff:.1f}%"
                    
                    # Position the annotation
                    y_pos = baseline * 1.1
                    ax.annotate(label, xy=(j, baseline), xytext=(j, y_pos),
                                ha='center', color=color, fontweight='bold')
        
        # Set titles and labels
        metric_labels = {
            'response_time': 'Average Response Time',
            'vm_count': 'Average VM Count',
            'error_budget': 'End Error Budget'
        }
        ax.set_title(metric_labels.get(metric, metric.replace('_', ' ').title()))
        ax.set_xlabel('')
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0 and baseline_results:  # Only add legend to the first subplot
            ax.legend()
    
    plt.tight_layout()
    return fig

def show_simulation_summary(results, pattern_name):
    """
    Display a summary table of simulation results with key metrics.
    """

    # Calculate key metrics
    avg_response_time = statistics.mean(results.response_history)
    avg_vm_count = statistics.mean(results.vm_history)
    max_vm_count = max(results.vm_history)
    min_error_budget = abs(min(results.error_budget_history))
    
    # Create a DataFrame for display
    summary_df = pd.DataFrame({
        'Pattern': [pattern_name],
        'Avg Response Time': [f"{avg_response_time:.2f}"],
        'Avg VM Count': [f"{avg_vm_count:.2f}"],
        'Max VM Count': [max_vm_count],
        'Min Error Budget': [f"{min_error_budget:.4f}"]
    })
    
    return summary_df.set_index('Pattern')

def display_decoded_predicates(scale_up_predicates_with_weights, scale_down_predicates_with_weights):
    """
    Display the decoded predicates in a human-readable format.
    """
    predicate_data = []
    
    # Helper function to process each predicate
    def process_predicate(predicate, weight, direction, index):
        pred_str = predicate.metric_fn.__name__
        # print(predicate.metric_fn.__name__)
        
        # Simple metric type determination
        if "cpu" in pred_str.lower():
            pred_type = "CPU Utilization"
        elif "response" in pred_str.lower():
            pred_type = "Response Time"
        elif "queue" in pred_str.lower():
            pred_type = "Queue Length"
        elif "budget" in pred_str.lower():
            pred_type = "Error Budget"
        else:
            pred_type = "Unknown"  # Default if not found
        
        # Determine the comparison operator
        operator = ">" if predicate.comparison_fn.__name__ == "gt" else "<"
        
        # Get threshold from string representation
        threshold = predicate.threshold
        
        return {
            "Direction": direction,
            "Index": index,
            "Metric": pred_type,
            "Operator": operator,
            "Threshold": f"{threshold:.2f}",
            "Weight": f"{weight:.1f}"
        }
    
    # Process scale-up predicates
    for i, (predicate, weight) in enumerate(scale_up_predicates_with_weights):
        predicate_data.append(process_predicate(predicate, weight, "Scale Up", i+1))
    
    # Process scale-down predicates
    for i, (predicate, weight) in enumerate(scale_down_predicates_with_weights):
        predicate_data.append(process_predicate(predicate, weight, "Scale Down", i+1))
    
    # Create and return the DataFrame
    return pd.DataFrame(predicate_data)


def create_metric_comparison_tables(results_list, baseline_results, traffic_patterns):
    """
    Creates tables for comparing GA and baseline metrics.
    """
 
    
    # Initialize data lists for each metric
    response_data = []
    vm_data = []
    error_data = []
    
    for pattern in traffic_patterns:
        # Get baseline metrics
        baseline = baseline_results[pattern]
        baseline_response = statistics.mean(baseline.response_history)
        baseline_vm = statistics.mean(baseline.vm_history)
        baseline_error = min(baseline.error_budget_history)
        
        # Get GA metrics (median across runs)
        ga_runs = [run[pattern] for run in results_list if pattern in run]
        
        
        ga_response = statistics.median([statistics.mean(run.response_history) for run in ga_runs])
        ga_vm = statistics.median([statistics.mean(run.vm_history) for run in ga_runs])
        ga_error = statistics.median([min(run.error_budget_history) for run in ga_runs])
        
        # Calculate differences
        response_diff = ((baseline_response - ga_response) / baseline_response) * 100
        vm_diff = ((baseline_vm - ga_vm) / baseline_vm) * 100
        error_diff = baseline_error - ga_error
        
        # Add to response data
        response_data.append({
            'Traffic Pattern': pattern,
            'Baseline avg response time': f"{baseline_response:.2f}",
            'GA median of avg response time': f"{ga_response:.2f}",
            'Improvement (%)': f"{response_diff:.2f}%"
        })
        
        # Add to VM data
        vm_data.append({
            'Traffic Pattern': pattern,
            'Baseline avg VM Count': f"{baseline_vm:.2f}",
            'GA median of avg VM Count': f"{ga_vm:.2f}",
            'Reduction (%)': f"{vm_diff:.2f}%"
        })
        
        # Add to error budget data
        error_data.append({
            'Traffic Pattern': pattern,
            'Baseline avg error budget': f"{baseline_error:.2f}",
            'GA median of avg error budget': f"{ga_error:.2f}",
            'Difference': f"{error_diff:.2f}"
        })
    
    # Create DataFrames
    response_df = pd.DataFrame(response_data)
    vm_df = pd.DataFrame(vm_data)
    error_df = pd.DataFrame(error_data)
    
    # Styling function for percentages (improvement is positive)
    def color_improvement(val):
        if isinstance(val, str) and "%" in val:
            try:
                pct = float(val.replace("%", ""))
                if pct > 0:
                    return 'color: green'
                elif pct < 0:
                    return 'color: red'
            except:
                pass
        return ''
    
    # Styling function for error budget values
    def color_error_budget(val):
        if isinstance(val, str):
            try:
                value = float(val)
                if value < 0:
                    return 'color: red'
                else:
                    return 'color: green'
            except:
                pass
        return ''
    
    # Styling function for error budget difference
    def color_error_diff(val):

        diff = float(val)
        if diff > 0:
            return 'color: red'  # Higher error budget is worse
        elif diff < 0:
            return 'color: green'  # Lower error budget is better

    # Style response time table
    response_styled = response_df.style.map(
        color_improvement, 
        subset=['Improvement (%)']
    ).set_caption("Response Time Comparison (lower is better)")
    
    # Style VM count table
    vm_styled = vm_df.style.map(
        color_improvement,
        subset=['Reduction (%)']
    ).set_caption("VM Count Comparison (lower is better)")
    
    # Style error budget table
    error_styled = error_df.style.map(
        color_error_budget,
        subset=['GA median of avg error budget', 'Baseline avg error budget']
    ).map(
        color_error_diff,
        subset=['Difference']
    ).set_caption("Error Budget Comparison (higher is better)")
    

    return response_styled, vm_styled, error_styled


