import wandb
import numpy as np 
import random
# from pdb import set_trace
from test_training import prepare_training_data
from mvnns_demand_query_training.mvnn_dq_training import train_mvnn_parallel_mixed
import torch 
import argparse 
import matplotlib.pyplot as plt 
import matplotlib


from pdb import set_trace


# wandb tracking
wandb_tracking = True 

bidder_type = 'regional'
instances_to_average = 5   # TODO: change back to 5

parser = argparse.ArgumentParser()

from jnius import autoclass
CPLEXUtils = autoclass('org.marketdesignresearch.mechlib.utils.CPLEXUtils')
SolveParams = autoclass('edu.harvard.econcs.jopt.solver.SolveParam')
CPLEXUtils.SOLVER.setSolveParam(SolveParams.THREADS,4)


#train parameter arguments
parser.add_argument("--l2_reg", help="l2 regularization", type=float, default= 1e-4)
parser.add_argument("--learning_rate", help="learning rate", default= 0.005, type=float)
parser.add_argument("--epochs", help="number of epochs", default= 15, type=int)   
parser.add_argument("--batch_size_dq", help="batch size", type=int, default = 1)

parser.add_argument("--max_linear_prices_multiplier", help="max linear prices multiplier", default= 2 * 3, type=float)  # NOTE: This used to be 2! 
parser.add_argument("--use_gradient_clipping", help="use gradient clipping", default= 'false', type=str)
parser.add_argument("--price_file_name", help="name of the price file, without domain", default= 'values_for_null_price_seeds1-100', type=str)

# MVNN parameter arguments
parser.add_argument("--num_hidden_layers", help="number of hidden layers", default= 1, type=int)
parser.add_argument("--num_hidden_units", help="number of hidden units", default= 20, type=int)
parser.add_argument("--dropout_prob", help="dropout probability", default= 0, type=float)
parser.add_argument("--lin_skip_connection", help="initialization method", default= 'false', type=str)

parser.add_argument("--scale_multiplier", help="scale multiplier, used only with dynamic scaling", default= 1, type=float)

# arguments for the training on value queries. 
parser.add_argument("--cached_solution_frequency", help="how often to re-calculate optimal solutions", default= 20, type=int)
parser.add_argument("--loss_vq", help="name of the price file, without domain", default= 'mse', type=str)
parser.add_argument("--batch_size_vq", help="batch size", type=int, default = 1)





def bidder_type_to_bidder_id(SATS_domain,
                             bidder_type):
    bidder_id_mappings = {'GSVM': {'national': [6], 'regional': [0, 1, 2, 3, 4, 5]},
                          'LSVM': {'national': [0], 'regional': [1, 2, 3, 4, 5]},
                          'SRVM': {'national': [5, 6], 'regional': [3, 4], 'high_frequency': [2], 'local': [0, 1]},
                          'MRVM': {'national': [7, 8, 9], 'regional': [3, 4, 5, 6], 'local': [0, 1, 2]}
                          }

    bidder_id = np.random.choice(bidder_id_mappings[SATS_domain][bidder_type], size=1, replace=False)[0]
    print(f'BIDDER ID:{bidder_id}')

    return bidder_id


# Define training function that takes in hyperparameter values from `wandb.config` and uses them to train a model and return metric
def main(): 

    # 0. Set random seeds
    torch.manual_seed(0)
    
    config = parser.parse_args()

    # 1. SATS Parameters
    # ----------------------------------------------
    SATS_parameters = {"SATS_domain": 'SRVM',
                    "isLegacy": False,
                    "SATS_seed": 1,
                        }
    # ----------------------------------------------

    # 2. Training Parameters
    # ----------------------------------------------
    TRAIN_parameters_single_bidder = {"number_train_data_points": 40,
                        "number_train_data_points_cardinal": 20,
                        "data_seed": 1,   # TODO: remove the / 10!!
                        "number_val_data_points": 200,   # TODO: Turn this back to 200  
                        "number_val_data_points_cardinal": 50000, #TODO: Turn this back to 50000
                        "val_points_multipliers": (0.75, 1.5), # NOTE: this only affects the PV dataset, not the generalization dataset
                        "instances_averaged": instances_to_average,
                        "max_linear_prices_multiplier": config.max_linear_prices_multiplier,  # sample from larger prices
                        "price_file_name": config.price_file_name,
                        'batch_size_dq': config.batch_size_dq,
                        'batch_size_vq': config.batch_size_vq,
                        'epochs': config.epochs,
                        'cached_solution_frequency': config.cached_solution_frequency,
                        'l2_reg': config.l2_reg,
                        'learning_rate': config.learning_rate,
                        'clip_grad_norm': 1,
                        'use_gradient_clipping': config.use_gradient_clipping,
                        'scale_multiplier': config.scale_multiplier,  # only used with dynamic scaling
                        'print_frequency': 1, 
                        'loss_vq': config.loss_vq,
                        }
    TRAIN_parameters = {f'Bidder_{bidder_id}': TRAIN_parameters_single_bidder for bidder_id in range(11)}
    # ----------------------------------------------

    # 3. MVNN Parameters
    # ----------------------------------------------
    MVNN_parameters_single_bidder = {'num_hidden_layers': config.num_hidden_layers,
                    'num_hidden_units': config.num_hidden_units,
                    'layer_type': 'MVNNLayerReLUProjected',
                    'target_max': 1, # TODO: check
                    'lin_skip_connection': config.lin_skip_connection, 
                    'dropout_prob': config.dropout_prob,
                    'init_method':'custom',
                    'random_ts': [0,1],
                    'trainable_ts': True,
                    'init_E': 1,
                    'init_Var': 0.09,
                    'init_b': 0.05,
                    'init_bias': 0.05,
                    'init_little_const': 0.1
                        }  
    MVNN_parameters = {f'Bidder_{bidder_id}': MVNN_parameters_single_bidder for bidder_id in range(11)}
    # ----------------------------------------------

    # 4. MIP Parameters
    # ----------------------------------------------
    MIP_parameters = {
        'timeLimit': 3600 * 10, # Default +inf
        'MIPGap': 1e-06, # Default 1e-04
        'IntFeasTol': 1e-8, # Default 1e-5
        'FeasibilityTol': 1e-9 # Default 1e-6
    }  
    # ----------------------------------------------

    # 4.5 Mechanism Parameters (only needed for CCA)
    # ----------------------------------------------
    mechanism_parameters = { 
                            "cca_start_linear_item_prices": np.load(f'{SATS_parameters["SATS_domain"]}_average_item_values_seeds_201-1200.npy'), # NOTE: only used for "initial_demand_query_method=cca"
                            "cca_initial_prices_multiplier": 0.2, # NOTE: only used for "initial_demand_query_method=cca", will multiply the initial prices. 
                            # "cca_increment" : 0.075,
                            "cca_increment" : 0.125, 
                            "dynamic_scaling": True if SATS_parameters["SATS_domain"] == 'MRVM' else False, 
                            }
    # ---------------------------------------------- 
    
    # 5. Run
    # ----------------------------------------------
    if wandb_tracking:
        run = wandb.init(project=f'SERVER-{SATS_parameters["SATS_domain"]}-HPO-Biddertype_{bidder_type}_mixed_v1.3', 
                        config={**SATS_parameters,**TRAIN_parameters,**MVNN_parameters},
                        reinit=True)
    

    # note: setting WANDB tracking to False in order to use an external tracker 
    # for hyperparameter optimization
    metrics_all_runs = [] 
    for i in range(instances_to_average):
        print(f'Starting run {i}')

        bidder_id = bidder_type_to_bidder_id(SATS_domain= SATS_parameters['SATS_domain'], bidder_type=bidder_type)
        print(f'Bidder id chosen: {bidder_id}')

        
        
        # set random seed
        seed_offset = i
        SATS_parameters['SATS_seed'] = SATS_parameters['SATS_seed'] + seed_offset
        TRAIN_parameters[f'Bidder_{bidder_id}']['data_seed'] = TRAIN_parameters[f'Bidder_{bidder_id}']['data_seed'] + seed_offset
        
        

        # --- prepare all the data --- # 
        # SATS_auction_instance, good_capacities, train_DQdata, train_VQdata, max_linear_prices, scales, GSVM_national_bidder_goods_of_interest 
        print('---> Right before preparing the data')
        
        (capacity_generic_goods, elicited_dqs, dqs_val_data, elicited_vqs, 
            vqs_val_data, scales, GSVM_national_bidder_goods_of_interest, 
            TRAIN_parameters_single_bidder, SATS_parameters, SATS_auction_instance) = prepare_training_data(SATS_parameters=SATS_parameters,
                                        TRAIN_parameters=TRAIN_parameters[f'Bidder_{bidder_id}'],
                                        init_method= 'cca', 
                                        MECHANISM_parameters= mechanism_parameters,
                                        init_method_vq= 'random',
                                        bidder_id=bidder_id)  
        

        print('---> Done preparing data!!!')
        TRAIN_parameters[f'Bidder_{bidder_id}'] =  TRAIN_parameters_single_bidder

        # --- inspect the demand queries --- # 
        requested_bundles = elicited_dqs[f'Bidder_{bidder_id}'][0]
        unique_requested_bundles = np.unique(requested_bundles, axis=0).shape[0]
        print(f'Unique requested bundles: {unique_requested_bundles}')
        # set_trace()
        
        result_dict = train_mvnn_parallel_mixed(bidder_id= bidder_id,
                                                capacity_generic_goods = capacity_generic_goods,
                                                elicited_dqs = elicited_dqs, 
                                                dqs_val_data = dqs_val_data,
                                                elicited_vqs = elicited_vqs,
                                                vqs_val_data = vqs_val_data,
                                                scales = scales, 
                                                SATS_parameters = SATS_parameters,
                                                TRAIN_parameters = TRAIN_parameters,
                                                MVNN_parameters = MVNN_parameters,
                                                MIP_parameters = MIP_parameters,
                                                GSVM_national_bidder_goods_of_interest = GSVM_national_bidder_goods_of_interest,
                                                num_cpu_per_job = 4,
                                                good_to_license_mapping = None, 
                                                SATS_auction_instance= SATS_auction_instance, 
                                                val_frequency= 20
                                                )  # we will use the generic approach so no need for this mapping

        
        metrics = result_dict[f'Bidder_{bidder_id}'][1]
        metrics_all_runs.append(metrics)

    epochs =  config.epochs 
    for epoch in range(epochs):
        if 'val_metrics_demand_queries' in metrics_all_runs[0][epoch].keys():
            avg_metrics_val_dqs = {k: np.mean([metrics_all_runs[j][epoch]['val_metrics_demand_queries'][k] for j in range(instances_to_average)]) for k in metrics_all_runs[0][epoch]['val_metrics_demand_queries'].keys()}
        else: 
            avg_metrics_val_dqs = None
        
        if 'val_metrics_value_queries' in metrics_all_runs[0][epoch].keys():
            avg_metrics_val_vqs = {k: np.mean([metrics_all_runs[j][epoch]['val_metrics_value_queries'][k] for j in range(instances_to_average)]) for k in metrics_all_runs[0][epoch]['val_metrics_value_queries'].keys()}
        else: 
            avg_metrics_val_vqs = None

        if 'train_metrics_vqs' in metrics_all_runs[0][epoch].keys():
            avg_metrics_train_vqs = {k: np.mean([metrics_all_runs[j][epoch]['train_metrics_vqs'][k] for j in range(instances_to_average)]) for k in metrics_all_runs[0][epoch]['train_metrics_vqs'].keys()}

        if 'train_scaled_dq_loss' in metrics_all_runs[0][epoch].keys():
            avg_train_scaled_dq_loss = np.mean([metrics_all_runs[j][epoch]['train_scaled_dq_loss'] for j in range(instances_to_average)])
        else:
            avg_train_scaled_dq_loss = None

        if 'train_scaled_vq_loss' in metrics_all_runs[0][epoch].keys():
            avg_train_scaled_vq_loss = np.mean([metrics_all_runs[j][epoch]['train_scaled_vq_loss'] for j in range(instances_to_average)])

        

        
        wandb_dict = {"epochs": epoch} 
        if avg_metrics_val_dqs is not None:
            new_dict = {
                'val_DQs_r2': avg_metrics_val_dqs['r2'],
                'val_DQs_r2_centered': avg_metrics_val_dqs['r2_centered'],
                'val_DQs_kendall_tau': avg_metrics_val_dqs['kendall_tau'],
                'val_DQs_mae': avg_metrics_val_dqs['mae'],
                'r2_train': avg_metrics_val_dqs['r2_train'],
                'r2_centered_train': avg_metrics_val_dqs['r2_centered_train'],
                'kendall_tau_train': avg_metrics_val_dqs['kendall_tau_train'],
                'mae_train': avg_metrics_val_dqs['mae_train'],
                'mae_scaled_train': avg_metrics_val_dqs['mae_scaled_train'],
                'mse_train': avg_metrics_val_dqs['mse_train'],
                'mse_scaled_train': avg_metrics_val_dqs['mse_scaled_train'],
            }

            wandb_dict.update(new_dict)

        if avg_metrics_val_vqs is not None:
            new_dict = {
                'val_VQs_r2': avg_metrics_val_vqs['r2'],
                'val_VQs_r2_centered': avg_metrics_val_vqs['r2_centered'],
                'val_VQs_kendall_tau': avg_metrics_val_vqs['kendall_tau'],
                'val_VQs_mae': avg_metrics_val_vqs['mae'],
                'val_VQs_mae_scaled': avg_metrics_val_vqs['mae_scaled'],
                'val_VQs_mse': avg_metrics_val_vqs['mse'],
            }
            wandb_dict.update(new_dict)



        if avg_train_scaled_dq_loss is not None:
            wandb_dict['train_scaled_dq_loss'] = avg_train_scaled_dq_loss
        if avg_train_scaled_vq_loss is not None:
            wandb_dict['train_scaled_vq_loss'] = avg_train_scaled_vq_loss




        # set_trace()
        if wandb_tracking:
            wandb.log(wandb_dict)
        
    metrics_single_run_last_epoch = metrics_all_runs[-1][epochs - 1]

    # Generalization performance on random bunbles 
    true_values = metrics_single_run_last_epoch['val_metrics_value_queries']['scaled_true_values']
    predicted_values = metrics_single_run_last_epoch['val_metrics_value_queries']['scaled_predicted_values']

    # Generalization performance on the training dataset for the demand query data
    # These bundles were created based on the CCA prices. 
    true_values_train = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_true_values_train']
    predicted_values_train = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_predicted_values_train']
    inferred_values_train = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_inferred_values_train']

    # Generalization performance on the validation dataset for the demand query data 
    # Those bundles were created by asking the bidder a demand query for random price vectors
    true_values_PV = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_true_values']
    predicted_values_PV = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_predicted_values']
    inferred_values_PV = metrics_single_run_last_epoch['val_metrics_demand_queries']['scaled_inferred_values']

    # Predictive performance on the VQ train dataset
    # set_trace()
    true_values_train_vqs = metrics_single_run_last_epoch['train_metrics_vqs']['true_values_train'].flatten()
    predicted_values_train_vqs = metrics_single_run_last_epoch['train_metrics_vqs']['predicted_values_train'].flatten()

    
    # Plot used up to this point 
    plt.figure(figsize=(9,9))
    plt.scatter(true_values, predicted_values, c='crimson', s = 0.4, alpha = 0.4,facecolors='none', edgecolors='r', label = 'Random Bundles Set')
    plt.scatter(true_values_train, predicted_values_train, c='blue', s = 7, alpha = 1,facecolors='none', edgecolors='b', label = 'Train (CCA) Set')
    plt.scatter(true_values_PV, predicted_values_PV, c='green', s = 7, alpha = 1,facecolors='none', edgecolors='g', label = 'Demand on Random Prices Set')
    plt.scatter(true_values_train, inferred_values_train, c='pink', s = 7, alpha = 1,facecolors='none', edgecolors='k', label = 'Inferred Values Train Set')
    plt.scatter(true_values_PV, inferred_values_PV, c='orange', s = 7, alpha = 1,facecolors='none', edgecolors='k', label = 'Inferred Values Demand Set')

    # the point of the last CCA iteration, closet to where the prices ended up
    plt.scatter([true_values_train[-1]], [predicted_values_train[-1]], c='pink', s = 30, alpha = 1,facecolors='none', edgecolors='k', label = 'Last CCA Iteration')


    plt.legend(loc='upper left', fontsize=15)

    # plt.yscale('log')
    # plt.xscale('log')

    p1 = max(max(predicted_values), max(true_values))
    p2 = min(min(predicted_values), min(true_values))
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')

    if wandb_tracking:
        wandb.log({"true_vs_predicted_plot" : wandb.Image(plt)})
        
        

    
    # --- INFORMS-style presentation, figure including everything ---
    
    plt.figure(figsize =(9,9))
    plt.scatter(true_values_train, inferred_values_train, s = 30, alpha = 1, marker = 'v',facecolors='none', edgecolors='lightgreen', label = 'I.V. for DQ trainset')
    plt.scatter(true_values_PV, inferred_values_PV,  s = 30, alpha = 0.8,facecolors='none', edgecolors='lightskyblue', marker='v', label = 'I.V. for DQ validation set')
    plt.scatter(true_values_train, predicted_values_train,  s = 30, alpha = 1,facecolors='none', edgecolors='darkgreen', label = 'NN P. for DQ trainset')
    plt.scatter(true_values_PV, predicted_values_PV, s = 30, alpha = 0.8,facecolors='none', edgecolors='royalblue', label = 'NN P. for DQ validation set')
    
    plt.scatter(true_values, predicted_values, c='crimson', s = 0.4, alpha = 0.4,facecolors='none', edgecolors='lightcoral', label = 'NN P. for VQ validation set')
    plt.scatter(true_values_train_vqs, predicted_values_train_vqs, s = 30, alpha = 1,facecolors='none', c = 'plum',edgecolors='plum', label = 'NN P. for VQ trainset')
    
    
    legend = plt.legend(loc='upper left', fontsize=14)

    # p1 = max(max(predicted_values), max(true_values))
    # p2 = min(min(predicted_values), min(true_values))
    p1 = 1
    p2 = 0
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15 + 5)
    plt.ylabel('Predictions', fontsize=15 + 5)
    # plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')
    plt.grid()

    # set both axes to start at zero 
    # plt.xlim(0, max(max(predicted_values), max(true_values)))
    # plt.ylim(0, max(max(predicted_values), max(true_values)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Get the handles (line2D instances) and labels from the legend
    handles, labels = legend.legendHandles, legend.get_texts()


    # set the right ticks, ensure that it is the right fonttype 
    plt.xticks([0,1])
    plt.yticks([0,1])
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    # Set the size of the symbols in the legend
    for handle in handles:
        handle.set_sizes([100])

    # Change tick sizes
    # Get the current axes
    ax = plt.gca()

    # Increase the tick sizes
    ax.tick_params(axis='both', which='both', labelsize=16, width=2, length=6)


    if wandb_tracking:
        wandb.log({"true_vs_predicted_plot_fancy" : wandb.Image(plt)})

    plt.figure(figsize =(9,9))
    # plt.scatter(true_values_train, inferred_values_train, s = 30, alpha = 1, marker = 'v',facecolors='none', edgecolors='lightgreen', label = 'I.V. for DQ trainset')
    plt.scatter(true_values_PV, inferred_values_PV,  s = 30, alpha = 0.8,facecolors='none', edgecolors='lightskyblue', marker='v', label = 'I.V. for DQ validation set')
    # plt.scatter(true_values_train, predicted_values_train,  s = 30, alpha = 1,facecolors='none', edgecolors='darkgreen', label = 'NN P. for DQ trainset')
    plt.scatter(true_values_PV, predicted_values_PV, s = 30, alpha = 0.8,facecolors='none', edgecolors='royalblue', label = 'NN P. for DQ validation set')
    
    plt.scatter(true_values, predicted_values, c='crimson', s = 0.4, alpha = 0.4,facecolors='none', edgecolors='lightcoral', label = 'NN P. for VQ validation set')
    # plt.scatter(true_values_train_vqs, predicted_values_train_vqs, s = 30, alpha = 1,facecolors='none', c = 'plum',edgecolors='plum', label = 'NN P. for VQ trainset')
    
    
    legend = plt.legend(loc='upper left', fontsize=14)

    # p1 = max(max(predicted_values), max(true_values))
    # p2 = min(min(predicted_values), min(true_values))
    p1 = 1
    p2 = 0
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15 + 5)
    plt.ylabel('Predictions', fontsize=15 + 5)
    # plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')
    plt.grid()

    # set both axes to start at zero 
    # plt.xlim(0, max(max(predicted_values), max(true_values)))
    # plt.ylim(0, max(max(predicted_values), max(true_values)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Get the handles (line2D instances) and labels from the legend
    handles, labels = legend.legendHandles, legend.get_texts()


    # set the right ticks, ensure that it is the right fonttype 
    plt.xticks([0,1])
    plt.yticks([0,1])
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    # Set the size of the symbols in the legend
    for handle in handles:
        handle.set_sizes([100])

    # Change tick sizes
    # Get the current axes
    ax = plt.gca()

    # Increase the tick sizes
    ax.tick_params(axis='both', which='both', labelsize=16, width=2, length=6)


    if wandb_tracking:
        wandb.log({"true_vs_predicted_plot_fancy_val_sets" : wandb.Image(plt)})
        # run.finish()


    plt.figure(figsize =(9,9))
    plt.scatter(true_values_train, inferred_values_train, s = 30, alpha = 1, marker = 'v',facecolors='none', edgecolors='lightgreen', label = 'I.V. for DQ trainset')
    # plt.scatter(true_values_PV, inferred_values_PV,  s = 30, alpha = 0.8,facecolors='none', edgecolors='lightskyblue', marker='v', label = 'I.V. for DQ validation set')
    plt.scatter(true_values_train, predicted_values_train,  s = 30, alpha = 1,facecolors='none', edgecolors='darkgreen', label = 'NN P. for DQ trainset')
    # plt.scatter(true_values_PV, predicted_values_PV, s = 30, alpha = 0.8,facecolors='none', edgecolors='royalblue', label = 'NN P. for DQ validation set')
    
    # plt.scatter(true_values, predicted_values, c='crimson', s = 0.4, alpha = 0.4,facecolors='none', edgecolors='lightcoral', label = 'NN P. for VQ validation set')
    plt.scatter(true_values_train_vqs, predicted_values_train_vqs, s = 30, alpha = 1,facecolors='none', c = 'plum',edgecolors='plum', label = 'NN P. for VQ trainset')
    
    
    legend = plt.legend(loc='upper left', fontsize=14)

    # p1 = max(max(predicted_values), max(true_values))
    # p2 = min(min(predicted_values), min(true_values))
    p1 = 1
    p2 = 0
    plt.plot([p1, p2], [p1, p2], ls="--", c=".3")
    plt.xlabel('True Values', fontsize=15 + 5)
    plt.ylabel('Predictions', fontsize=15 + 5)
    # plt.title('True vs. Predicted Values', fontsize=15)
    plt.axis('equal')
    plt.grid()

    # set both axes to start at zero 
    # plt.xlim(0, max(max(predicted_values), max(true_values)))
    # plt.ylim(0, max(max(predicted_values), max(true_values)))
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Get the handles (line2D instances) and labels from the legend
    handles, labels = legend.legendHandles, legend.get_texts()


    # set the right ticks, ensure that it is the right fonttype 
    plt.xticks([0,1])
    plt.yticks([0,1])
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


    # Set the size of the symbols in the legend
    for handle in handles:
        handle.set_sizes([100])

    # Change tick sizes
    # Get the current axes
    ax = plt.gca()

    # Increase the tick sizes
    ax.tick_params(axis='both', which='both', labelsize=16, width=2, length=6)


    if wandb_tracking:
        wandb.log({"true_vs_predicted_plot_fancy_train_sets" : wandb.Image(plt)})
        run.finish()


main()

# # Start sweep agent with the function defined above
# wandb.agent(sweep_id, function=main, count=2)