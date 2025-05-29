import wandb
import tqdm
import os
import pickle
from pdb import set_trace


def extract_and_filter_mlcca_data(entity, project, config_filter=None, metric_filter=None, x_axis="Clock Round", number_of_goods=42):
    """
    Filter runs in a W&B project based on configuration and metric criteria, and extract ML-CCA data.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    metric_filter (tuple): A tuple specifying the metric name and the condition (metric_name, comparison_operator, value).
    metric (str): The metric to extract (default is "efficiency").
    x_axis (str): The x-axis parameter (default is "Clock Round").
    number_of_goods (int): Number of goods in the auction.
    
    Returns:
    dict: A dictionary where keys are run IDs and values are lists of (x_axis, metrics) pairs.
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")

    # Initialize a dictionary to store the filtered data
    filtered_data = {}

    # Initialize a counter to keep track of the number of runs that matched the configuration filter
    matched_runs = 0

    # Loop over each run and apply filters
    # use tqdm to show progress bar
    for run in tqdm.tqdm(runs):
        # Apply configuration filter
        if config_filter:
            match = all(run.config.get(k) == v for k, v in config_filter.items())
            if not match:
                # print which condition of the filter failed 
                # print('Skippig run as config filter failed with run values:', [run.config.get(k) for k in config_filter.keys()])
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

        # If all filters pass, extract the relevant data
        metrics = [
            "Found Clearing Prices",
            "Efficiency Clock Bids Per Clock Round",
            "Efficiency Raised Clock Bids Per Clock Round"
        ]
        all_metrics = [x_axis] + metrics
        history = run.history(keys=all_metrics)
        
        # # Filter "Found Clearing Prices" to exclude rounds <= 20
        # clearing_prices = history["Found Clearing Prices"].apply(lambda x: x if history[x_axis] > 20 else None)
        
        filtered_data[sats_seed] = list(zip(
            history[x_axis],
            history["Found Clearing Prices"],
            history["Efficiency Clock Bids Per Clock Round"],
            history["Efficiency Raised Clock Bids Per Clock Round"]
        ))

    if matched_runs != 100:
        print(f'Got {matched_runs} runs that matched the configuration filter instead of 100, check the configuration filter.')
    print('Returning results from {} runs that matched the configuration filter.'.format(matched_runs))
    return filtered_data


def extract_and_filter_mlcca_data(entity, project, config_filter=None, metric_filter=None, x_axis="Clock Round", number_of_goods=42):
    """
    Filter runs in a W&B project based on configuration and metric criteria, and extract ML-CCA data.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    metric_filter (tuple): A tuple specifying the metric name and the condition (metric_name, comparison_operator, value).
    metric (str): The metric to extract (default is "efficiency").
    x_axis (str): The x-axis parameter (default is "Clock Round").
    number_of_goods (int): Number of goods in the auction.
    
    Returns:
    dict: A dictionary where keys are run IDs and values are lists of (x_axis, metrics) pairs.
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")

    # Initialize a dictionary to store the filtered data
    filtered_data = {}

    # Initialize a counter to keep track of the number of runs that matched the configuration filter
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

        # If all filters pass, extract the relevant data
        metrics = [
            "Found Clearing Prices",
            "Efficiency Clock Bids Per Clock Round",
            "Efficiency Raised Clock Bids Per Clock Round"
        ]
        all_metrics = [x_axis] + metrics
        history = run.history(keys=all_metrics)
        
        # Set "Found Clearing Prices" to zero for rounds <= 20
        clearing_prices = [0 if round_num <= 20 else price for round_num, price in zip(history[x_axis], history["Found Clearing Prices"])]
        
        filtered_data[sats_seed] = list(zip(
            history[x_axis],
            clearing_prices,
            history["Efficiency Clock Bids Per Clock Round"],
            history["Efficiency Raised Clock Bids Per Clock Round"]
        ))

    if matched_runs != 100:
        print(f'Got {matched_runs} runs that matched the configuration filter instead of 100, check the configuration filter.')
    print('Returning results from {} runs that matched the configuration filter.'.format(matched_runs))
    return filtered_data


def extract_efficiency_data(entity, project, config_filter=None, x_axis="Clock Round"):
    """
    Extract the efficiency data (Efficiency Clock Bids Per Clock Round and Efficiency Raised Clock Bids Per Clock Round) 
    for all clock rounds and return it in a dictionary.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    x_axis (str): The x-axis parameter (default is "Clock Round").
    
    Returns:
    dict: A dictionary where keys are SATS_seed and values are lists of (x_axis, Efficiency Clock Bids Per Clock Round, Efficiency Raised Clock Bids Per Clock Round).
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")

    # Initialize a dictionary to store the filtered data
    filtered_data = {}
    matched_runs = 0

    # Loop over each run and apply filters
    for run in tqdm.tqdm(runs):
        # Apply configuration filter
        if config_filter:
            match = all(run.config.get(k) == v for k, v in config_filter.items())
            if not match:
                continue

        # Extract SATS_seed from configuration
        sats_seed = run.config.get("SATS_seed")
        if sats_seed is None:
            print(f"Skipping run {run.id} as SATS_seed is not found in configuration")
            continue

        matched_runs += 1

        # Extract the relevant efficiency data
        metrics = [
            "Efficiency Clock Bids Per Clock Round",
            "Efficiency Raised Clock Bids Per Clock Round"
        ]
        all_metrics = [x_axis] + metrics
        history = run.history(keys=all_metrics)
        # history = run.history(keys=metrics)

        # check for empty dataframe
        if history.empty:
            print(f"Skipping run {run.id} as history is empty")
            set_trace()

        # set_trace()
        filtered_data[sats_seed] = list(zip(
            history[x_axis],
            history["Efficiency Clock Bids Per Clock Round"],
            history["Efficiency Raised Clock Bids Per Clock Round"]
        ))

    if matched_runs != 100:
        print(f'Got {matched_runs} runs that matched the configuration filter instead of 100, check the configuration filter.')
        raise ValueError('Number of runs that matched the configuration filter is not 100.')
    return filtered_data


# def extract_efficiency_data(entity, project, config_filter=None, x_axis="Elicited Bids"):
#     """
#     Extract the efficiency data (Efficiency Clock Bids Per Clock Round and Efficiency Raised Clock Bids Per Clock Round) 
#     for all clock rounds and return it in a dictionary.
#     This is for the ML-CCA runs in the new framework that have Elicited Bids in the x-axis instead of Clock Round.
    
#     Parameters:
#     entity (str): The W&B entity (username or team name).
#     project (str): The W&B project name.
#     config_filter (dict): A dictionary specifying configuration parameters to filter by.
#     x_axis (str): The x-axis parameter (default is "Clock Round").
    
#     Returns:
#     dict: A dictionary where keys are SATS_seed and values are lists of (x_axis, Efficiency Clock Bids Per Clock Round, Efficiency Raised Clock Bids Per Clock Round).
#     """
#     # Authenticate with W&B
#     api = wandb.Api()

#     # Access the specified project
#     runs = api.runs(f"{entity}/{project}")

#     # Initialize a dictionary to store the filtered data
#     filtered_data = {}
#     matched_runs = 0

#     # Loop over each run and apply filters
#     for run in tqdm.tqdm(runs):
#         # Apply configuration filter
#         if config_filter:
#             match = all(run.config.get(k) == v for k, v in config_filter.items())
#             if not match:
#                 continue

#         # Extract SATS_seed from configuration
#         sats_seed = run.config.get("SATS_seed")
#         if sats_seed is None:
#             print(f"Skipping run {run.id} as SATS_seed is not found in configuration")
#             continue

#         matched_runs += 1

#         # Extract the relevant efficiency data
#         metrics = [
#             "Efficiency Clock Bids Per Clock Round",
#             "Efficiency Raised Clock Bids Per Clock Round"
#         ]
#         all_metrics = [x_axis] + metrics
#         history = run.history(keys=all_metrics)

#         filtered_data[sats_seed] = list(zip(
#             history[x_axis],
#             history["Efficiency Clock Bids Per Clock Round"],
#             history["Efficiency Raised Clock Bids Per Clock Round"]
#         ))

#     if matched_runs != 100:
#         print(f'Got {matched_runs} runs that matched the configuration filter instead of 100, check the configuration filter.')
#         raise ValueError('Number of runs that matched the configuration filter is not 100.')
#     return filtered_data


def extract_clearing_price_data(entity, project, config_filter=None, x_axis="Clock Round"):
    """
    Extract the clearing price data for the last clock round and return a dictionary indicating 
    whether clearing prices were found in the last round for each SATS_seed.
    
    Parameters:
    entity (str): The W&B entity (username or team name).
    project (str): The W&B project name.
    config_filter (dict): A dictionary specifying configuration parameters to filter by.
    x_axis (str): The x-axis parameter (default is "Clock Round").
    
    Returns:
    dict: A dictionary where keys are SATS_seed and values are boolean (True if clearing prices were found in the last round, False otherwise).
    """
    # Authenticate with W&B
    api = wandb.Api()

    # Access the specified project
    runs = api.runs(f"{entity}/{project}")
    matched_runs = 0 

    # Initialize a dictionary to store the clearing price information
    clearing_price_data = {}

    # Loop over each run and apply filters
    for run in tqdm.tqdm(runs):
        # Apply configuration filter
        if config_filter:
            match = all(run.config.get(k) == v for k, v in config_filter.items())
            if not match:
                continue

        # Extract SATS_seed from configuration
        sats_seed = run.config.get("SATS_seed")
        if sats_seed is None:
            print(f"Skipping run {run.id} as SATS_seed is not found in configuration")
            continue

        matched_runs += 1

        # Extract the relevant clearing price data
        history = run.history(keys=[x_axis, "Found Clearing Prices"])

        # Check if "Found Clearing Prices" were logged in the last round
        if history.empty:
            clearing_price_data[sats_seed] = False
        else:
            last_round = history[x_axis].max()
            # set_trace()
            clearing_price_data[sats_seed] = history[history[x_axis] == last_round]['Found Clearing Prices'].values[0]

            # clearing_price_data[sats_seed] = not history[history[x_axis] == last_round]["Found Clearing Prices"].isna().all()

    if matched_runs != 100:
        print(f'Got {matched_runs} runs that matched the configuration filter instead of 100, check the configuration filter.')
        raise ValueError('Number of runs that matched the configuration filter is not 100.')
    return clearing_price_data


if __name__ == "__main__":
    entity = "anonymized"  # Replace with your W&B username or team name
    domain = 'MRVM'
    mechanism_to_extract = 'ML-CCA'
    
    
    if domain in ['GSVM', 'LSVM']:
        version = '1.5'
        if mechanism_to_extract == 'ML-CCA':
            config_filter = {'new_query_option': 'gd_linear_prices_on_W_v3', 'Qmax': 100, 'Qinit': 20, 'W_v3_feasibility_multiplier': 2, 
                            'hpo_file_name': 'hpo_configs.json', 'initial_demand_query_method': 'cca'}
        elif mechanism_to_extract == 'CCA':
            config_filter = {'initial_demand_query_method': 'cca', 'new_query_option': 'cca', 'Qmax': 100, 'Qinit': 20, 'hpo_file_name': 'hpo_configs.json'}
    
    elif domain in ['SRVM', 'MRVM']:
        version = '1.7'
        if mechanism_to_extract == 'ML-CCA':
            config_filter = {'new_query_option': 'gd_linear_prices_on_W_v3', 'Qmax': 100, 'Qinit': 20, 'W_v3_feasibility_multiplier': 2, 
                            'hpo_file_name': 'hpo_configs.json', 'initial_demand_query_method': 'cca'}
            if domain == 'MRVM': 
                config_filter['Qinit'] = 50 # Qinit for MRVM
        elif mechanism_to_extract == 'CCA':
            config_filter = {'initial_demand_query_method': 'cca', 'new_query_option': 'cca', 'Qmax': 100, 'Qinit': 20,  'hpo_file_name': 'hpo_configs.json'}
        
        
    metric_filter = None

    project = f'MLCCA_Domain_{domain}_v{version}'  # Replace with your W&B project name

    # Extract data for ML-CCA
    # filtered_mlcca_data = extract_and_filter_mlcca_data(entity, project, config_filter, metric_filter)
    filtered_mlcca_efficiency_data = extract_efficiency_data(entity, project, config_filter)
    filtered_mlcca_clearing_price_data = extract_clearing_price_data(entity, project, config_filter)
    # set_trace()

    # Save the extracted data
    save_path = './mlcca_results'
    if mechanism_to_extract == 'ML-CCA':
        filename_efficiency = f'mlcca_data_{domain}.pkl'
        filename_clearing_price = f'mlcca_clearing_price_data_{domain}.pkl'
    elif mechanism_to_extract == 'CCA':
        filename_efficiency = f'cca_data_{domain}.pkl'
        filename_clearing_price = f'cca_clearing_price_data_{domain}.pkl'
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename_efficiency), 'wb') as f:
        pickle.dump(filtered_mlcca_efficiency_data, f)
    with open(os.path.join(save_path, filename_clearing_price), 'wb') as f:
        pickle.dump(filtered_mlcca_clearing_price_data, f)
