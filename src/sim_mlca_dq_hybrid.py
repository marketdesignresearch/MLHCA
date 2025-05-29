# Libs
import argparse
import json
import logging
import os
import re
import numpy as np

# Own Libs
from mlca_demand_queries.mlca_hybrid_mechanism import hybrid_mechanism
from util import StreamToLogger
# from pdb import set_trace

# in order to properly parallelize on the server side
# --------------------------------------
from pysats import PySats

# set jnius classpath before doing anything else
PySats.getInstance()
from jnius import autoclass

from pysats_ext import GenericWrapper

CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,8)    # TODO: Change the number of threads to be the same as in the script.    
# --------------------------------------

def main(domain, init_method, new_query_option, cca_dq, mlcca_dq, mlcca_vq, forbid_individual_bundles, seed):
    
    # 0. Set up logging
    logging.basicConfig(level=logging.INFO)

    
    # 1. SATS Parameters
    # ----------------------------------------------
    SATS_parameters = {"SATS_domain": domain,
                       "isLegacy": False,
                       "SATS_seed": seed
                        }
    # ----------------------------------------------

    # 1.5 Load the json with the HPOed parameters for the MVNNs and the one for the linear prices
    hpo_file_name = 'hpo_configs_hybrid_medfast.json' # TODO: change back    
    hpo_dict =  json.load(open(hpo_file_name, 'r'))
    hpo_file_name_dqs = 'hpo_configs.json'    # the HPO file name for dqs     
    hpo_dict_dqs = json.load(open(hpo_file_name_dqs, 'r'))
    price_file_name = 'values_for_null_price_seeds1-100'
    average_price_file_name = 'average_item_values_seeds_201-1200'
    price_dict =  json.load(open(f'{domain}_{price_file_name}.json', 'r')) # AVG value per item   
    all_bidders_scales = {} 
    all_bidders_max_linear_prices = {}
    for key in price_dict.keys():
        if 'max_value_per_item' in key:
            id = int(re.findall(r'\d+', key)[0])
            all_bidders_max_linear_prices[id] = price_dict[key]['mean']
        if 'max_value' in key and 'per_item' not in key:
            id = int(re.findall(r'\d+', key)[0])
            all_bidders_scales[id] = price_dict[key]['mean'] 
    # ----------------------------------------------

    MVNN_parameters_all_bidders = {}
    TRAIN_parameters_all_bidders = {}

    MVNN_parameters_all_bidders_dqs = {}
    TRAIN_parameters_all_bidders_dqs = {}

    if domain == 'GSVM':
        num_bidders = 7
        num_items = 18
    elif domain == 'LSVM':
        num_bidders = 6
        num_items = 18
    elif domain == 'MRVM':
        num_bidders = 10
        num_items = 42
    # 'SRVM': {'national': [5, 6], 'regional': [3, 4], 'high_frequency': [2], 'local': [0, 1]},
    elif domain == 'SRVM':
        num_bidders = 7
        num_items = 3

    for i in range(num_bidders):
        if domain == 'GSVM':
            if i == 6:
                bidder_type = 'National'
            else:
                bidder_type = 'Regional'
        elif domain == 'LSVM':
            if i == 0: 
                bidder_type = 'National'
            else:
                bidder_type = 'Regional'
        elif domain in ['SRVM', 'SRVM_non_generic']:
            if i in [0, 1]: 
                bidder_type = 'Local'
            elif i in [2]:
                bidder_type = 'High Frequency'
            elif i in [3, 4]:
                bidder_type = 'Regional'
            elif i in [5, 6]:
                bidder_type = 'National'
        elif domain in ['MRVM', 'MRVM_non_generic']:
            if i in [0, 1, 2]: 
                bidder_type = 'Local'
            elif i in [3, 4, 5, 6]:
                bidder_type = 'Regional'
            elif i in [7, 8, 9]:
                bidder_type = 'National'
        
        # 2. MVNN Parameters
        # ----------------------------------------------
        MVNN_parameters = {'num_hidden_layers': hpo_dict[domain][bidder_type]['num_hidden_layers'],
                            'num_hidden_units': hpo_dict[domain][bidder_type]['num_hidden_units'],
                            'layer_type': 'MVNNLayerReLUProjected',
                            'target_max': 1, 
                            'lin_skip_connection': hpo_dict[domain][bidder_type]['lin_skip_connection'],
                            'dropout_prob': 0,
                            'init_method':'custom',
                            'random_ts': [0,1],
                            'trainable_ts': True,
                            'init_E': 1,
                            'init_Var': 0.09,
                            'init_b': 0.05,
                            'init_bias': 0.05,
                            'init_little_const': 0.1
                            }   
        MVNN_parameters_dqs = {'num_hidden_layers': hpo_dict_dqs[domain][bidder_type]['num_hidden_layers'],
                            'num_hidden_units': hpo_dict_dqs[domain][bidder_type]['num_hidden_units'],
                            'layer_type': 'MVNNLayerReLUProjected',
                            'target_max': 1, 
                            'lin_skip_connection': hpo_dict_dqs[domain][bidder_type]['lin_skip_connection'],
                            'dropout_prob': 0,
                            'init_method':'custom',
                            'random_ts': [0,1],
                            'trainable_ts': True,
                            'init_E': 1,
                            'init_Var': 0.09,
                            'init_b': 0.05,
                            'init_bias': 0.05,
                            'init_little_const': 0.1
                            } 
        # 3. Train Parameters
        # ----------------------------------------------
        if domain == 'GSVM':
            end_linear_item_prices_multiplier = 30 
        elif domain == 'LSVM':
            end_linear_item_prices_multiplier = 40 
        elif domain == 'MRVM':
            end_linear_item_prices_multiplier = 40 # TODO: FIX if we implement this method
        elif domain == 'SRVM':
            end_linear_item_prices_multiplier = 40 # TODO: FIX if we implement this method
        else: 
            end_linear_item_prices_multiplier = 40 # TODO: FIX if we implement this method
        

        TRAIN_parameters = {"number_val_data_points": 10,
                            "max_linear_prices_multiplier": hpo_dict[domain][bidder_type]['max_linear_prices_multiplier'], # NOTE: only used for "initial_demand_query_method=random"; they actually get loaded from the file
                            "max_linear_price": all_bidders_max_linear_prices[i], # NOTE: only used for "initial_demand_query_method=random"; they actually get loaded from the file
                            "scale": all_bidders_scales[i], # NOTE: they actually get loaded from the file
                            "start_linear_item_prices": np.zeros(num_items), # NOTE: only used for "initial_demand_query_method=increasing" 
                            "end_linear_item_prices": np.ones(num_items)* end_linear_item_prices_multiplier, # NOTE: only used for "initial_demand_query_method=increasing"
                            "price_file_name": 'values_for_null_price_seeds1-100.json',
                            "average_price_file_name": average_price_file_name,
                            'batch_size_dq': 1,
                            'batch_size_vq': hpo_dict[domain][bidder_type]['batch_size_vq'],
                            'loss_vq': hpo_dict[domain][bidder_type]['loss_vq'],  
                            'epochs': hpo_dict[domain][bidder_type]['epochs'],      # TODO: Change back to original!!! 
                            'cached_solution_frequency': hpo_dict[domain][bidder_type]['cached_solution_frequency'], 
                            'l2_reg': hpo_dict[domain][bidder_type]['l2_reg'],
                            'learning_rate': hpo_dict[domain][bidder_type]['learning_rate'],
                            'clip_grad_norm': 1,
                            'use_gradient_clipping': hpo_dict[domain][bidder_type]['use_gradient_clipping'],
                            'scale_multiplier': hpo_dict[domain][bidder_type].get('scale_multiplier', 1),    # NOTE: only used for dynamic scaling.
                            'print_frequency': 1,
                            'generic_approach': True # NOTE: should be true for the main table results 
                            }
        
        TRAIN_parameters_dqs = {"number_val_data_points": 10,
                            "max_linear_prices_multiplier": hpo_dict_dqs[domain][bidder_type]['max_linear_prices_multiplier'], # NOTE: only used for "initial_demand_query_method=random"; they actually get loaded from the file
                            "max_linear_price": all_bidders_max_linear_prices[i], # NOTE: only used for "initial_demand_query_method=random"; they actually get loaded from the file
                            "scale": all_bidders_scales[i], # NOTE: they actually get loaded from the file
                            "start_linear_item_prices": np.zeros(num_items), # NOTE: only used for "initial_demand_query_method=increasing" 
                            "end_linear_item_prices": np.ones(num_items)* end_linear_item_prices_multiplier, # NOTE: only used for "initial_demand_query_method=increasing"
                            "price_file_name": 'values_for_null_price_seeds1-100.json',
                            "average_price_file_name": average_price_file_name,
                            'batch_size_dq': 1,
                            'batch_size_vq': 1,  # NOTE: does not matter
                            'loss_vq': "MSE", # NOTE: does not matter
                            'epochs': hpo_dict_dqs[domain][bidder_type]['epochs'],      # TODO: Change back to original!!! 
                            'cached_solution_frequency': 1, 
                            'l2_reg': hpo_dict_dqs[domain][bidder_type]['l2_reg'],
                            'learning_rate': hpo_dict_dqs[domain][bidder_type]['learning_rate'],
                            'clip_grad_norm': 1,
                            'use_gradient_clipping': False,
                            'scale_multiplier': hpo_dict_dqs[domain][bidder_type].get('scale_multiplier', 1),    # NOTE: only used for dynamic scaling.
                            'print_frequency': 1,
                            'generic_approach': True # NOTE: should be true for the main table results 
                            }
        # ----------------------------------------------
        
        MVNN_parameters_all_bidders[f'Bidder_{i}'] = MVNN_parameters
        TRAIN_parameters_all_bidders[f'Bidder_{i}'] = TRAIN_parameters
        MVNN_parameters_all_bidders_dqs[f'Bidder_{i}'] = MVNN_parameters_dqs
        TRAIN_parameters_all_bidders_dqs[f'Bidder_{i}'] = TRAIN_parameters_dqs
    # ----------------------------------------------

    # 4. SET Mechanism PARAMETERS:
    mechanism_parameters = {
                            'Qinit': cca_dq,  # TODO: CHANGE THIS BACK TO cca_dq: These are the CCA queries that are made at the beginning of the auction.
                            'MLCCA-DQs': mlcca_dq,  # these are the MLCA demand queries, that are made during the auction, so we have a total of Qinit + MLCA-DQs queries up to this point
                            'MLCCA-VQs': mlcca_vq,  # these are the MLCA value queries, that are made during the auction, so we have a total of Qinit + MLCA-DQs + MLCCA-VQs queries up to this point
                            # 'Qinit': 10,
                            # 'MLCCA-DQs': 1,
                            # 'MLCCA-VQs': 6,
                            'new_query_option': new_query_option, 
                            'initial_demand_query_method': init_method, # select between 'random', 'increasing', 'cca', and 'cca_original'
                            'value_query_method': 'mlca', # select between 'mlca' and other future options  
                            'calculate_raised_bids': False, # if true: at every iteration the raised bids will be calculated and both efficiencies will be logged. 
                            "cca_start_linear_item_prices": np.load(f'{domain}_{average_price_file_name}.npy'), # NOTE: only used for "initial_demand_query_method=cca"
                            "cca_initial_prices_multiplier": 0.2 if domain in ['LSVM', 'MRVM'] else 1.6, # NOTE: only used for "initial_demand_query_method=cca", will multiply the initial prices. 
                            "calculate_profit_max_bids": False, # NOTE: This will calculate profit max bids for every ML-powered clock round, very expensive.
                            "calculate_profit_max_bids_unraised": False, 
                            "calculate_profit_max_bids_specific_rounds": [],    
                            "profit_max_grid": [],   # TODO: change back to [5, 100]
                            'parallelize_training': True,
                            'calculate_efficiency_per_iteration': True, 
                            'dynamic_scaling': False if domain in ['GSVM', 'LSVM', 'SRVM'] else True,   # only true for MRVM  
                            'hpo_file_name': hpo_file_name,
                            'W_epochs': 100 if domain in ['GSVM', 'LSVM'] else 300,  # NOTE: used to be 1000 for other domains for ML-CCA
                            'W_lr': 1 if domain in ['GSVM', 'LSVM'] else 4 if domain in ['SRVM'] else 5 * 1000000,
                            'W_lr_decay': 0.99 if domain in ['GSVM', 'LSVM', 'SRVM'] else 0.999, 
                            'W_v2_max_steps_without_improvement': 100 if domain in ['GSVM', 'LSVM', 'SRVM'] else 300,   # parameters for the new GD procedure to minimize W
                            'W_v2_lr': 1 if domain in ['GSVM', 'LSVM'] else 4 if domain in ['SRVM'] else 5 * 1000000,
                            'W_v2_lr_decay': 0.99 if domain in ['GSVM', 'LSVM', 'SRVM'] else 0.999, 
                            'W_v3_max_steps_without_improvement': 250,   # parameters for the new GD procedure to minimize W
                            'W_v3_max_steps': 150,   #  NOTE: used to be 300 for ML-CCA 
                            'W_v3_lr': 0.01, 
                            'W_v3_lr_decay': 0.995, 
                            'W_v3_filter_feasible': True,  # if True: GD on W will return the minimum price vector that lead to a predicted FEASIBLE allocation, when possible. 
                            'W_v3_feasibility_multiplier': 2,  # punish over demand during GD by more than under-demand, because the first one leads to infeasible bids, which are harder to combine.
                            'W_v3_feasibility_multiplier_increase_factor': 1.01,
                            'W_log_frequency': 10, # how often to log all details of W minimization, if wandb tracking is on.
                            'identical_p_threshold': 0.05, 
                            'identical_p_threshold_decay': 0.95,
                            'forbid_individual_bundles': forbid_individual_bundles,  # if true: the mechanism will forbid allocations on a per bundle basis, otherwise it will forbid full allocations.
                            'forbid_single_bidder': True,  # if true: the mechanism will solve a MILP for each bidder, only forbidding that bidder getting any of his previous bundles (the other bidders are free to get the bundle of a previous allocation)
                            'forbid_single_bidder_vq_threshold': 0, # the round after which the mechanism will start forbidding single bidders, if the above is true 
                            'marginal_economies': True, # if true: the mechanism will also query marginal economies in the VQ rounds
                            'marginal_economies_per_round': 3, # how many marginal economies to query per VQ round
                            'bridge_bid': True,   # if true: the mechanism will ask a single bridge bid between the DQs and the VQs 
                            'use_dq_HPO': True,  # if true: the mechanism will use the specialized MVNNs on DQs during the ML-CCA rounds 
                            }
    
    if mechanism_parameters['new_query_option'] == 'cca':
        if domain in ['LSVM', 'MRVM', 'SRVM']: 
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2
        elif domain in ['GSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 1.6
    elif mechanism_parameters['new_query_option'] in ['gd_linear_prices_on_W', 'gd_linear_prices_on_W_v2', 'gd_linear_prices_on_W_v3']:
        if domain in ['LSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.7
        elif domain in ['GSVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 1.6
        elif domain in ['MRVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2
        elif domain in ['SRVM']:
            mechanism_parameters['cca_initial_prices_multiplier'] = 0.2

    # -------------------
    if mechanism_parameters['new_query_option'] == 'cca':  
        if mechanism_parameters['Qmax'] == 100: 
            mechanism_parameters['cca_increment'] = 0.05
        elif mechanism_parameters['Qmax'] == 50:
            mechanism_parameters['cca_increment'] = 0.1025 # so that after 50 clock rounds we have the same possible price increments as in the 100 clock round case.
    elif mechanism_parameters['Qinit'] == 20:
        if domain in ['GSVM', 'LSVM', 'SRVM']:
            mechanism_parameters['cca_increment'] = 0.15  
        else: 
            mechanism_parameters['cca_increment'] = 0.22  
    elif mechanism_parameters['Qinit'] == 50:
        mechanism_parameters['cca_increment'] = 0.08
    elif mechanism_parameters['Qinit'] == 70:
        mechanism_parameters['cca_increment'] = 0.0565
    elif mechanism_parameters['Qinit'] == 60:
        mechanism_parameters['cca_increment'] = 0.0665
    else: 
        mechanism_parameters['cca_increment'] = 0.15
    

    # 5. SET MIP PARAMETERS:
    MIP_parameters = {
        'timeLimit': 3600 * 1.5, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }
    # -------------------

    # Create directory for results
    # --------------------------------------
    if mechanism_parameters['initial_demand_query_method'] == 'random':
        path_addition = f'max_linear_prices_multiplier_{TRAIN_parameters_all_bidders["Bidder_0"]["max_linear_prices_multiplier"]}_'
    elif mechanism_parameters['initial_demand_query_method'] == 'increasing':
        path_addition = f'start_linear_item_prices_{TRAIN_parameters_all_bidders["Bidder_0"]["start_linear_item_prices"][0]}_end_linear_item_prices_{TRAIN_parameters_all_bidders["Bidder_0"]["end_linear_item_prices"][0]}_'
    elif mechanism_parameters['initial_demand_query_method'] == 'cca':
        path_addition = f'cca_initial_prices_multiplier_{mechanism_parameters["cca_initial_prices_multiplier"]}_increment_{mechanism_parameters["cca_increment"]}_'
    res_path = os.path.join(os.getcwd(),
                            'results',
                             f'{domain}_qinit_{mechanism_parameters["Qinit"]}_initial_demand_query_method_{mechanism_parameters["initial_demand_query_method"]}_{path_addition}new_query_option_{mechanism_parameters["new_query_option"]}',
                             f'ML_config_hpo1',  # so that we know with which config and init price method the results were generated 
                             str(seed)
                             )
    

    os.makedirs(res_path, exist_ok=True)
    # --------------------------------------




    # 6. Run mechanism
    hybrid_mechanism(SATS_parameters = SATS_parameters,
              TRAIN_parameters = TRAIN_parameters_all_bidders,
              MVNN_parameters =MVNN_parameters_all_bidders,
              TRAIN_parameters_dqs= TRAIN_parameters_all_bidders_dqs,
              MVNN_parameters_dqs= MVNN_parameters_all_bidders_dqs,
              mechanism_parameters = mechanism_parameters,
              MIP_parameters = MIP_parameters,
              res_path = res_path, 
              wandb_tracking = True,  # TODO: Switch to true 
              wandb_project_name = f"MLCCA_Hybrid_Domain_{domain}_v1.8"   # TODO: Change back to no test 
            #   wandb_project_name = f"MLCCA_Domain_{domain}_2.0"   # TODO: Change back to no test 
              )
    # -------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", help="name of the domain to run", default= 'SRVM', type=str)
    parser.add_argument("--init_method", help="method for the Qinit queries, options: random, increasing, cca", default= 'cca', type=str)
    parser.add_argument("--cca_dq", help="number of initial queries", default= 20, type=int)
    parser.add_argument("--mlcca_dq", help="number of demand queries", default= 20, type=int)  # TODO: Change back to 20 
    parser.add_argument("--mlcca_vq", help="number of value queries", default= 60, type=int)  # TODO: Change back to 60 
    parser.add_argument("--seed", help="auction instance seed to run", default= 184, type=int)
    parser.add_argument("--new_query_option", help="new query option", default= 'gd_linear_prices_on_W_v3', type=str)  # options: gd_linear_prices_on_W, gd_linear_prices_on_W_v2, gd_linear_prices_on_W_v3, cca and gd_linear_prices_on_W_v3_cheating
    parser.add_argument("--forbid_individual_bundles", help="forbid individual bundles", default= 'true', type=str)

    args = parser.parse_args()
    domain = args.domain
    seed = args.seed
    init_method = args.init_method
    cca_dq = args.cca_dq
    mlcca_dq = args.mlcca_dq    
    mlcca_vq = args.mlcca_vq
    new_query_option = args.new_query_option
    forbid_individual_bundles = args.forbid_individual_bundles.lower() == 'true'

    print(f'Starting instance on domain: {domain} with init method: {init_method} cca_dq: {cca_dq} mlcca_dq: {mlcca_dq} mlcca_vq: {mlcca_vq} seed: {seed} new_query_option: {new_query_option}')
    
    
    main(domain=domain, init_method = init_method, new_query_option = new_query_option, 
         cca_dq= cca_dq, mlcca_dq=mlcca_dq, mlcca_vq=mlcca_vq, forbid_individual_bundles = forbid_individual_bundles, seed=seed)


