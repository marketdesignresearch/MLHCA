import wandb
import tqdm 
import os 
import pickle
from pdb import set_trace   

def extract_and_filter_efficiency_data(entity, project, config_filter=None, metric_filter=None, metric="Efficiency", 
        x_axis="Number of Elicited Bids", possible_skips = False):
    """
    Filter runs in a W&B project based on configuration and metric criteria, and extract efficiency data.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    metric_filter (tuple): A tuple specifying the metric name and the condition (metric_name, comparison_operator, value).
    metric (str): The metric to extract (default is "efficiency").
    x_axis (str): The x-axis parameter (default is "Number of Elicited Bids").
    possible_skips (bool): If True, allow for runs to be skipped if they do not have a history. If False, raise an error if a run is skipped. Useful for example for relative revenue where cleared runs have not logged revenue
    
    Returns:
    dict: A dictionary where keys are run IDs and values are lists of (x_axis, metric) pairs.
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")

    # Initialize a dictionary to store the filtered data
    filtered_data = {}
    matched_runs = 0

    # Loop over each run and apply filters
    # use tqdm to show progress bar
    for run in tqdm.tqdm(runs):
        # Apply configuration filter
        if config_filter:
            match = all(run.config.get(k) == v for k, v in config_filter.items())
            if not match:
                continue

        # Apply metric filter
        if metric_filter:
            metric_name, comparison_operator, threshold_value = metric_filter
            history = run.history(keys=[metric_name])
            metric_values = history[metric_name].dropna()
            
            # Define comparison operation
            if comparison_operator == "gt":
                if not any(metric_values > threshold_value):
                    continue
            elif comparison_operator == "lt":
                if not any(metric_values < threshold_value):
                    continue
            elif comparison_operator == "eq":
                if not any(metric_values == threshold_value):
                    continue
            else:
                raise ValueError("Invalid comparison operator. Use 'gt', 'lt', or 'eq'.")
            
        matched_runs += 1
            
        # Extract SATS_seed from configuration
        sats_seed = run.config.get("SATS_seed")
        if sats_seed is None:
            # Skip this run if SATS_seed is not found
            print(f"Skipping run {run.id} as SATS_seed is not found in configuration")
            continue

        # If all filters pass, extract efficiency data
        history = run.history(keys=[x_axis, metric])
        if history.empty:
            if possible_skips:
                print(f"Skipping run {run.id} as history is empty")
                # set_trace()
                continue
            else:
                raise ValueError(f"Run {run.id} has empty history, even though it passed all filters")

        filtered_data[sats_seed] = list(zip(history[x_axis], history[metric]))

    if matched_runs != 50: 
        raise ValueError(f"Matched runs is {matched_runs} instead of 50")

    return filtered_data


def extract_and_filter_price_data(entity, project, config_filter=None, metric_filter=None, x_axis="Number of Elicited Bids", number_of_goods = 42):
    """
    Filter runs in a W&B project based on configuration and metric criteria, and extract efficiency data.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    metric_filter (tuple): A tuple specifying the metric name and the condition (metric_name, comparison_operator, value).
    metric (str): The metric to extract (default is "efficiency").
    x_axis (str): The x-axis parameter (default is "Number of Elicited Bids").
    
    Returns:
    dict: A dictionary where keys are run IDs and values are lists of (x_axis, metric) pairs.
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")

    # Initialize a dictionary to store the filtered data
    filtered_data = {}

    # Loop over each run and apply filters
    # use tqdm to show progress bar
    for run in tqdm.tqdm(runs):
        # Apply configuration filter
        if config_filter:
            config_filter_results = [run.config.get(k) for k in config_filter.keys()]
            # set_trace()
            match = all(run.config.get(k) == v for k, v in config_filter.items())
            if not match:
                # show which filter failed
                print(f"Skipping run {run.id} as config filter failed: {config_filter_results}")
                continue

        # Apply metric filter
        if metric_filter:
            metric_name, comparison_operator, threshold_value = metric_filter
            history = run.history(keys=[metric_name])
            metric_values = history[metric_name].dropna()
            
            # Define comparison operation
            if comparison_operator == "gt":
                if not any(metric_values > threshold_value):
                    continue
            elif comparison_operator == "lt":
                if not any(metric_values < threshold_value):
                    continue
            elif comparison_operator == "eq":
                if not any(metric_values == threshold_value):
                    continue
            else:
                raise ValueError("Invalid comparison operator. Use 'gt', 'lt', or 'eq'.")
            
        # Extract SATS_seed from configuration
        sats_seed = run.config.get("SATS_seed")
        if sats_seed is None:
            # Skip this run if SATS_seed is not found
            print(f"Skipping run {run.id} as SATS_seed is not found in configuration")
            continue

        

        # If all filters pass, extract price data
        metrics = [f'Extended Prices Good {i}' for i in range(number_of_goods)]
        all_metrics = [x_axis] + metrics
        history = run.history(keys=all_metrics)
        # set_trace()
        filtered_data[sats_seed] = list(zip(history[x_axis], [history[metric] for metric in metrics]))

    return filtered_data

if __name__ == "__main__":
    entity = "cerg"  # Replace with your W&B username or team name
    domain = 'LSVM'
    # if domain == 'SRVM':
    #     version = 1.8
    # else:
    #     version = 1.7
    version = 1.7
    project = f'MLCCA_Hybrid_Domain_{domain}_v{version}'  # Replace with your W&B project name  # TODO: change back to 1.7 

    config_filter = {'forbid_single_bidder': True, 'forbid_single_bidder_vq_threshold': 0, 'bridge_bid': True}
    
    
    # config_filter = {'forbid_single_bidder': "true"}
    # config_filter = None
    
    if domain == 'SRVM':
        number_of_goods = 3
    elif domain == 'MRVM':
        number_of_goods = 42
    elif domain in ['GSVM', 'LSVM']:
        number_of_goods = 18
    else:
        raise ValueError(f"Domain {domain} not recognized")


    metric_filter = None

    # --- FOR THE PRICE DATA ---  #
    # filtered_price_data = extract_and_filter_price_data(entity, project, config_filter, metric_filter, number_of_goods=number_of_goods)
    # set_trace()

    # # # save the results in the hybrid results folder
    # save_path = './hybrid_results'
    # filename = f'price_data_{domain}_v1.7.pkl'  # no need to save extra details in the file name as the price data do not depend on the configuration
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, filename), 'wb') as f:
    #     pickle.dump(filtered_price_data, f)

    # # --- FOR THE EFFICIENCY DATA ---  #
    # filtered_efficiency_data = extract_and_filter_efficiency_data(entity, project, config_filter, metric_filter)
    # set_trace()
    # save_path = './hybrid_results'
    # if config_filter['bridge_bid']:
    #     filename = f'efficiency_data_{domain}_v{version}_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}.pkl'
    # else:
    #     filename = f'efficiency_data_{domain}_v{version}_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}_no_bridge.pkl'
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, filename), 'wb') as f:
    #     pickle.dump(filtered_efficiency_data, f)


    # # --- FOR THE INFERRED SCW DATA ---  #
    # filtered_inferred_scw_data = extract_and_filter_efficiency_data(entity, project, config_filter, metric_filter, metric= "Inferred SCW")
    # # set_trace()
    # save_path = './hybrid_results'
    # if config_filter['bridge_bid']:
    #     filename = f'inferred_scw_data_{domain}_v1.7_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}.pkl'
    # else:
    #     filename = f'inferred_scw_data_{domain}_v1.7_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}_no_bridge.pkl'
    # os.makedirs(save_path, exist_ok=True)
    # with open(os.path.join(save_path, filename), 'wb') as f:
    #     pickle.dump(filtered_inferred_scw_data, f)

    # # --- FOR THE Revenue DATA ---  #
    filtered_revenue_data = extract_and_filter_efficiency_data(entity, project, config_filter, metric_filter, metric= "Relative Revenue", possible_skips= True)
    # set_trace()
    save_path = './hybrid_results'
    if config_filter['bridge_bid']:
        filename = f'revenue_data_{domain}_v{version}_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}.pkl'
    else:
        filename = f'revenue_data_{domain}_v{version}_forbid_single_bidder_{config_filter["forbid_single_bidder"]}_threshold_{config_filter["forbid_single_bidder_vq_threshold"]}_no_bridge.pkl'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), 'wb') as f:
        pickle.dump(filtered_revenue_data, f)
