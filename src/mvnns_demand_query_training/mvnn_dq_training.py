# Libs
import numpy as np
import torch
import sklearn.metrics
from scipy import stats as scipy_stats
import wandb
import time
import logging
import torch.nn.functional as F
# from pdb import set_trace

# Own Libs
from mvnns.mvnn import MVNN
from mvnns.mvnn_generic import MVNN_GENERIC
from milps.gurobi_mip_mvnn_single_bidder_util_max import GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX
from milps.gurobi_mip_mvnn_generic_single_bidder_util_max import GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX 

from mvnns.explicit_100_UB_mvnn_including_dqs import Explicit100PercentUpperBoundIncludingDQS
from mvnns.losses import NOMU_loss_hybrid

from pysats import PySats
PySats.getInstance()
from pysats_ext import GenericWrapper


# from pdb import set_trace
#%%
def dq_train_mvnn_helper(model,
                        optimizer,
                        use_gradient_clipping,
                        clip_grad_norm,
                        train_loader_demand_queries,
                        SATS_domain,
                        bidder_id,
                        GSVM_national_bidder_goods_of_interest,
                        device,
                        MIP_parameters,
                        non_generic_approach = False, 
                        return_cached_dataloader = False
                        ):
    
    print('DEBUG: called dq_train_mvnn')
    model.train()
    loss_dq_list = []
    triplets_list = [] # will store demand_vector, price_vector, predicted_demand for each sample

    for batch_idx, (demand_vector, price_vector) in enumerate(train_loader_demand_queries):
        price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
        optimizer.zero_grad()

        #--------------------------------
        # IMPORTANT: we need to transform the weights and the biases of the MVNN to be non-positive and non-negative, respectively.
        model.transform_weights()
        #--------------------------------

        # compute the network's predicted answer to the demand query
        if SATS_domain in ['GSVM', 'LSVM'] or non_generic_approach: 
            solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(model=model,
                                                            SATS_domain = SATS_domain,
                                                            bidder_id = bidder_id,
                                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
            solver.generate_mip(price_vector.numpy()[0])
        elif SATS_domain in ['SRVM', 'MRVM']: 
            solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=model)  # if the domain is generic -> use the new MIP 
            solver.generate_mip(price_vector.numpy()[0])
        else:
            raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')
        
        try: 
            predicted_demand = solver.solve_mip(outputFlag=False,
                                                verbose = False,
                                                timeLimit = MIP_parameters["timeLimit"],
                                                MIPGap = MIP_parameters["MIPGap"],
                                                IntFeasTol = MIP_parameters["IntFeasTol"],
                                                FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                )
            predicted_demand = np.array(predicted_demand)
        except:
            print('--- MIP is unbounded, skipping this sample! ---')
            continue 

        if return_cached_dataloader:
            triplets_list.append((demand_vector.cpu().numpy(), price_vector.cpu().numpy(), predicted_demand))

        # get the predicted value for that answer
        predicted_value = model(torch.from_numpy(predicted_demand).float())

        predicted_utility = predicted_value - torch.dot(price_vector.flatten(), torch.from_numpy(predicted_demand).float())

        # get the predicted utility for the actual demand vector
        predicted_value_at_true_demand = model(demand_vector)

        predicted_utility_at_true_demand = predicted_value_at_true_demand - torch.dot(price_vector.flatten(), demand_vector.flatten())


        # compute the loss
        predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
        if predicted_utility_difference < 0:
            print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')

        loss = torch.relu(predicted_utility_difference)   # for numerical stability
        loss_dq_list.append(loss.detach().numpy())
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

    if return_cached_dataloader:
        # Convert lists of numpy arrays to a single numpy array before creating tensors
        demand_vectors = np.array([t[0] for t in triplets_list])
        price_vectors = np.array([t[1] for t in triplets_list])
        predicted_demands = np.array([t[2] for t in triplets_list])

        # Create tensors from numpy arrays
        demand_tensor = torch.tensor(demand_vectors, dtype=torch.float32)
        price_tensor = torch.tensor(price_vectors, dtype=torch.float32)
        predicted_demand_tensor = torch.tensor(predicted_demands, dtype=torch.float32)

        # triplets_dataset = torch.utils.data.TensorDataset(torch.tensor([t[0] for t in triplets_list]).float(), 
        #                                  torch.tensor([t[1] for t in triplets_list]).float(), 
        #                                  torch.tensor([t[2] for t in triplets_list]).float())
        triplets_dataset = torch.utils.data.TensorDataset(demand_tensor, price_tensor, predicted_demand_tensor)
        triplets_dataloader = torch.utils.data.DataLoader(triplets_dataset, batch_size=train_loader_demand_queries.batch_size, shuffle=True)
        
        return np.mean(loss_dq_list), triplets_dataloader

    return np.mean(loss_dq_list)


def dq_train_mvnn_helper_cached_dataloader(model,
                        optimizer,
                        use_gradient_clipping,
                        clip_grad_norm,
                        train_loader_demand_queries_cached,
                        device
                        ):
    """
    Performs the same training as dq_train_mvnn_helper, but with cached predicted demands for the given prices, 
    instead of envoking Gurobi to re-compute the optimal demand for each price vector.
    """

    model.train()
    loss_dq_list = []

    for batch_idx, (demand_vector, price_vector, cached_predicted_demand) in enumerate(train_loader_demand_queries_cached):
        price_vector, demand_vector, cached_predicted_demand = price_vector.to(device), demand_vector.to(device), cached_predicted_demand.to(device)
        optimizer.zero_grad()

        #--------------------------------
        # IMPORTANT: we need to transform the weights and the biases of the MVNN to be non-positive and non-negative, respectively.
        model.transform_weights()
        #--------------------------------

        # get the predicted value for the cached predicted demand
        predicted_value = model(cached_predicted_demand)

        predicted_utility = predicted_value - torch.dot(price_vector.flatten(), cached_predicted_demand.flatten())

        # get the predicted utility for the actual demand vector
        predicted_value_at_true_demand = model(demand_vector)

        predicted_utility_at_true_demand = predicted_value_at_true_demand - torch.dot(price_vector.flatten(), demand_vector.flatten())


        # compute the loss
        predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand

        loss = torch.relu(predicted_utility_difference)   # for numerical stability
        loss_dq_list.append(loss.detach().numpy())
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()



    return np.mean(loss_dq_list)


def vq_train_mvnn_helper(model,
                         optimizer,
                         use_gradient_clipping,
                         clip_grad_norm,
                         train_loader_value_queries,
                         device,
                         loss_type):
    """
    A helper function to train a model on supplementary round bids 

    Args:
        model: The neural network model.
        optimizer: The optimizer used for training.
        use_gradient_clipping (bool): Whether to use gradient clipping.
        clip_grad_norm (float): The max norm for gradient clipping.
        train_loader_value_queries: DataLoader for the training data.
        device: The device to train on (CPU or GPU).
        loss_type (str): The type of loss function to use ('mse' or 'mae').
    """

    model.train()
    loss_sr_list = []
    true_values = []
    predicted_values = []

    for batch_idx, (bundle_vector, bid) in enumerate(train_loader_value_queries):
        bundle_vector, bid = bundle_vector.to(device), bid.to(device).unsqueeze(1)  # Ensure bid has shape [1, 1]
        optimizer.zero_grad()

        #--------------------------------
        # IMPORTANT: we need to transform the weights and the biases of the MVNN to be non-positive and non-negative, respectively.
        model.transform_weights()
        #--------------------------------

        # compute the network's predicted bid to the value query
        predicted_value = model(bundle_vector.float())

        # compute the loss
        if loss_type == 'mse':
            loss = F.mse_loss(predicted_value, bid)
        elif loss_type == 'mae':
            loss = F.l1_loss(predicted_value, bid)
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        loss_sr_list.append(loss.detach().cpu().numpy())
        loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        
        true_values_array = bid.detach().cpu().numpy()
        predicted_values_array = predicted_value.detach().cpu().numpy()
        true_values_flattened = true_values_array.flatten().tolist()
        predicted_values_flattened = predicted_values_array.flatten().tolist()

        true_values += true_values_flattened
        predicted_values += predicted_values_flattened

    try: 
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
    except:
        print('ERROR: could not convert true and predicted values to numpy arrays')
        print(f'Predicted values: {predicted_values}')
        print(f'True values: {true_values}')
        # set_trace()
        true_values = np.array([0,0])
        predicted_values = np.array([0,0])
        
    return np.mean(loss_sr_list), {'true_values_train': true_values, 'predicted_values_train': predicted_values}


def nomu_hybrid_train_mvnn_helper(ub_model,
                        mean_model,
                        exp_upper_bound_net,
                        NOMU_optimizer,
                        use_gradient_clipping,
                        clip_grad_norm,
                        train_loader_value_queries,
                        device,
                        NOMU_parameters):
    """
    A helper function to train a UB model on supplementary round bids 

    Args:
        ub_model: The UB neural network model to train. 
        mean_model: The trained mean model.
        exp_upper_bound_net: The explicit upper bound network.
        NOMU_optimizer: The optimizer used for training.
        use_gradient_clipping (bool): Whether to use gradient clipping.
        clip_grad_norm (float): The max norm for gradient clipping.
        train_loader_value_queries: DataLoader for the training data.
        device: The device to train on (CPU or GPU).
        NOMU_parameters: The parameters for the NOMU loss function.
    """

    ub_model.train()
    loss_sr_list = []
    true_values = []
    predicted_values = []

    for batch_idx, (bundle_vector, bid) in enumerate(train_loader_value_queries):
        bundle_vector, bid = bundle_vector.to(device), bid.to(device).unsqueeze(1)  # Ensure bid has shape [1, 1]
        NOMU_optimizer.zero_grad()

        #--------------------------------
        # IMPORTANT: we need to transform the weights and the biases of the NOMU-MVNN to be non-positive and non-negative, respectively.
        ub_model.transform_weights()
        #--------------------------------

        # compute the network's predicted bid to the value query
        predicted_value = ub_model(bundle_vector.float())

        # compute the loss
        nomu_loss = NOMU_loss_hybrid(ub_output=predicted_value,
            X_train = bundle_vector,
            target = bid,
            pi_sqr = NOMU_parameters['pi_sqr'],
            pi_exp = NOMU_parameters['pi_exp'],
            pi_above_mean = NOMU_parameters['pi_above_mean'],
            c_exp = NOMU_parameters['c_exp'],
            n_aug = NOMU_parameters['n_aug'],
            mean_model = mean_model,
            ub_model = ub_model,
            exp_upper_bound_net = exp_upper_bound_net)


        loss_sr_list.append(nomu_loss.detach().cpu().numpy())
        nomu_loss.backward()

        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(ub_model.parameters(), clip_grad_norm)
        NOMU_optimizer.step()

        
        true_values_array = bid.detach().cpu().numpy()
        predicted_values_array = predicted_value.detach().cpu().numpy()
        true_values_flattened = true_values_array.flatten().tolist()
        predicted_values_flattened = predicted_values_array.flatten().tolist()

        true_values += true_values_flattened
        predicted_values += predicted_values_flattened

    try: 
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)
    except:
        print('ERROR: could not convert true and predicted values to numpy arrays')
        print(f'Predicted values: {predicted_values}')
        print(f'True values: {true_values}')
        # set_trace()
        true_values = np.array([0,0])
        predicted_values = np.array([0,0])
        
    return np.mean(loss_sr_list), {'true_values_train': true_values, 'predicted_values_train': predicted_values}


def dq_val_mvnn(trained_model,
                val_loader,
                train_loader,
                SATS_auction_instance,
                bidder_id,
                scale,
                SATS_domain,
                GSVM_national_bidder_goods_of_interest,
                device,
                MIP_parameters
                ):
    """
    Validates an MVNN on a demand query validation set.
    """
    
    
    trained_model.eval()
    val_metrics = {}

    scaled_value_preds = []
    demand_vectors = []
    price_vectors = []
    with torch.no_grad():
        for demand_vector, price_vector in val_loader:
            price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
            scaled_value_prediction = trained_model(demand_vector)
            scaled_value_preds.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
            demand_vectors.extend(demand_vector.cpu().numpy().tolist())
            price_vectors.extend(price_vector.cpu().numpy().tolist())

    

    scaled_value_preds = np.array(scaled_value_preds)
    true_values = np.array([SATS_auction_instance.calculate_value(bidder_id, demand_vector) for demand_vector in demand_vectors])
    scaled_true_values = true_values/scale

    inferred_values = np.array([np.dot(price_vector, demand_vector) for (price_vector, demand_vector) in zip(price_vectors, demand_vectors)])

    value_preds = scaled_value_preds * scale


    common_scale = np.mean(true_values)
    common_scale_true_values = true_values / common_scale
    common_scale_value_preds = value_preds / common_scale

    # 1. generalization performance measures (on the validation set, that is drawn using price vectors)
    # --------------------------------------
    val_metrics['r2'] = sklearn.metrics.r2_score(y_true=true_values, y_pred= value_preds)  # This is R2 coefficient of determination
    val_metrics['kendall_tau'] = scipy_stats.kendalltau(scaled_value_preds, scaled_true_values).correlation
    val_metrics['mae'] = sklearn.metrics.mean_absolute_error(value_preds, true_values)
    val_metrics['mae_scaled'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds, common_scale_true_values)
    val_metrics['r2_centered'] = sklearn.metrics.r2_score(y_true=true_values - np.mean(true_values), y_pred= value_preds - np.mean(value_preds)) 
    # a centered R2, because constant shifts in model predictions should not really affect us  

    val_metrics['scaled_true_values'] = scaled_true_values  # also store all true /predicted values so that we can make true vs predicted plots
    val_metrics['scaled_predicted_values'] = scaled_value_preds
    val_metrics['scaled_inferred_values'] = inferred_values

    # set_trace()



    # --------------------------------------
    # 1.5 If the training loader is given -> also measure predictive performance on that 
    if train_loader is not None:
        scaled_value_preds_train = []
        demand_vectors_train = []
        price_vectors_train = []
        with torch.no_grad():
            for demand_vector, price_vector in train_loader:
                price_vector, demand_vector = price_vector.to(device), demand_vector.to(device)
                scaled_value_prediction = trained_model(demand_vector)
                scaled_value_preds_train.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
                demand_vectors_train.extend(demand_vector.cpu().numpy().tolist())
                price_vectors_train.extend(price_vector.cpu().numpy().tolist())
    
        scaled_value_preds_train = np.array(scaled_value_preds_train)
        true_values_train = np.array([SATS_auction_instance.calculate_value(bidder_id, demand_vector) for demand_vector in demand_vectors_train])

        scaled_true_values_train = true_values_train/scale
        value_preds_train = scaled_value_preds_train * scale

        inferred_values_train = np.array([np.dot(price_vector, demand_vector) for (price_vector, demand_vector) in zip(price_vectors_train, demand_vectors_train)])

        common_scale = np.mean(true_values_train)
        common_scale_true_values_train = true_values_train / common_scale
        common_scale_value_preds_train = value_preds_train / common_scale
        
        val_metrics['r2_train'] = sklearn.metrics.r2_score(y_true=true_values_train, y_pred=value_preds_train)
        val_metrics['r2_centered_train'] = sklearn.metrics.r2_score(y_true=true_values_train - np.mean(true_values_train), y_pred= value_preds_train - np.mean(value_preds_train))
        val_metrics['kendall_tau_train'] = scipy_stats.kendalltau(true_values_train, value_preds_train).correlation
        val_metrics['mae_train'] = sklearn.metrics.mean_absolute_error(true_values_train, value_preds_train)
        val_metrics['mae_scaled_train'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds_train, common_scale_true_values_train)
        val_metrics['mse_train'] = sklearn.metrics.mean_squared_error(true_values_train, value_preds_train)
        val_metrics['mse_scaled_train'] = sklearn.metrics.mean_squared_error(common_scale_value_preds_train, common_scale_true_values_train)

        # also store all true /predicted values so that we can make true vs predicted plots
        val_metrics['scaled_true_values_train'] = scaled_true_values_train
        val_metrics['scaled_predicted_values_train'] = scaled_value_preds_train
        val_metrics['scaled_inferred_values_train'] = inferred_values_train

        
        print(f'Predictive performance on training set. R2: {val_metrics["r2_train"]}, R2C: {val_metrics["r2_centered_train"]} Kendall Tau: {val_metrics["kendall_tau_train"]}, MAE: {val_metrics["mae_train"]}')

    
    # 2. DQ loss performance measure (same as training loss)
    # --------------------------------------
    
    # Create the common MVNN MIP solver
    if SATS_domain in ['GSVM', 'LSVM']:
        solver = GUROBI_MIP_MVNN_SINGLE_BIDDER_UTIL_MAX(model=trained_model,
                                                        SATS_domain = SATS_domain,
                                                        bidder_id = bidder_id,
                                                        GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest)
        solver.generate_mip(price_vector.numpy()[0])
    elif SATS_domain in ['SRVM', 'MRVM']:
        solver = GUROBI_MIP_MVNN_GENERIC_SINGLE_BIDDER_UTIL_MAX(model=trained_model)
        solver.generate_mip(price_vector.numpy()[0])
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')
    
    val_dq_loss = 0 
    predicted_demands = [] 
    for (j, price_vector) in enumerate(price_vectors): 
        # update the prices in the MIP  objective to the price vector of the current datapoint
        solver.update_prices_in_objective(price_vector)

        # compute the network's predicted answer to the demand query
    
        try: 
            predicted_demand = solver.solve_mip(outputFlag=False,
                                                verbose = False,
                                                timeLimit = MIP_parameters["timeLimit"],
                                                MIPGap = MIP_parameters["MIPGap"],
                                                IntFeasTol = MIP_parameters["IntFeasTol"],
                                                FeasibilityTol = MIP_parameters["FeasibilityTol"],
                                                )
            predicted_demand = np.array(predicted_demand)
        except:
            print('MIP is unbounded, something is wrong!')
            predicted_demand = np.ones(demand_vector.shape[0])

        predicted_demands.append(predicted_demand)

        # get the predicted value for that answer
        predicted_value = trained_model(torch.from_numpy(predicted_demand).float()).item()

        predicted_utility = predicted_value - np.dot(price_vector, predicted_demand)

        # get the predicted utility for the actual demand vector
        demand_vector = demand_vectors[j]
        predicted_value_at_true_demand = scaled_value_preds[j]

        predicted_utility_at_true_demand = predicted_value_at_true_demand - np.dot(price_vector, demand_vector)

        # compute the loss
        predicted_utility_difference = predicted_utility - predicted_utility_at_true_demand
        val_dq_loss = val_dq_loss + predicted_utility_difference
        if predicted_utility_difference < 0:
            print(f'predicted utility difference is negative: {predicted_utility_difference}, something is wrong!')
            # solver._print_info() # NOTE: if you print info on a MIP that is unbounded, it will crush... 


    val_metrics['scaled_dq_loss'] = val_dq_loss / len(price_vectors)
    # --------------------------------------


    # 3. Regret performance measure 
    # --------------------------------------
    regret = 0
    for (j, price_vector) in enumerate(price_vectors):
        # calculate the optimal true utility for the true demand vector
        scaled_true_value =  scaled_true_values[j]
        scaled_true_opt_utility = scaled_true_value - np.dot(price_vector, demand_vectors[j])

        # calculate the true utility for the predicted demand vector
        predicted_demand = predicted_demands[j]
        scaled_value_at_predicted_demand = SATS_auction_instance.calculate_value(bidder_id, predicted_demand) / scale
        scaled_utility_at_predicted_demand = scaled_value_at_predicted_demand - np.dot(price_vector, predicted_demand)

        regret = regret + (scaled_true_opt_utility - scaled_utility_at_predicted_demand)


    val_metrics['mean_regret'] = (regret * scale) / len(price_vectors)
    val_metrics['mean_regret_scaled'] = val_metrics['mean_regret'] / common_scale # regret scaled by the common scale of the generalization set to make numbers interpretable
    # --------------------------------------
    return val_metrics


def vq_val_mvnn(trained_model,
                val_loader,
                scale,
                device,
                ):
    """
    Validate an MVNN on a value query validation set.
    params:
    trained_model: The trained MVNN model.
    val_loader: The validation data loader. Requires the data to be 
    """
    
    trained_model.eval()
    val_metrics = {}

    scaled_value_preds = []
    scaled_true_values = []
    bundle_vectors = []
    with torch.no_grad():
        for bundle_vector, true_value in val_loader:
            bundle_vector, true_value = bundle_vector.to(device), true_value.to(device)
            scaled_value_prediction = trained_model(bundle_vector)
            scaled_value_preds.extend(scaled_value_prediction.cpu().numpy().flatten().tolist())
            bundle_vectors.extend(bundle_vector.cpu().numpy().tolist())
            scaled_true_values.extend(true_value.cpu().numpy().tolist())

    scaled_value_preds = np.array(scaled_value_preds)
    scaled_true_values = np.array(scaled_true_values)



    value_preds = scaled_value_preds * scale
    true_values = np.array(scaled_true_values) * scale

    common_scale = np.mean(true_values)

    common_scale_true_values = true_values / common_scale
    common_scale_value_preds = value_preds / common_scale

    # 1. generalization performance measures (on the validation set, that is drawn using the mlca uniform bundle distribution)
    # --------------------------------------
    val_metrics['r2'] = sklearn.metrics.r2_score(y_true=true_values, y_pred= value_preds)  # This is R2 coefficient of determination
    val_metrics['kendall_tau'] = scipy_stats.kendalltau(scaled_value_preds, scaled_true_values).correlation
    val_metrics['mae'] = sklearn.metrics.mean_absolute_error(value_preds, true_values)
    val_metrics['mae_scaled'] = sklearn.metrics.mean_absolute_error(common_scale_value_preds, common_scale_true_values)
    val_metrics['mse'] = sklearn.metrics.mean_squared_error(value_preds, true_values)
    val_metrics['r2_centered'] = sklearn.metrics.r2_score(y_true=true_values - np.mean(true_values), y_pred= value_preds - np.mean(value_preds)) 
    # a centered R2, because constant shifts in model predictions should not really affect us  

    val_metrics['scaled_true_values'] = scaled_true_values  # also store all true /predicted values so that we can make true vs predicted plots
    val_metrics['scaled_predicted_values'] = scaled_value_preds


    # also store all true /predicted values so that we can make true vs predicted plots
    # val_metrics['scaled_true_values_train'] = scaled_true_values_train
    # val_metrics['scaled_predicted_values_train'] = scaled_value_preds_train

    return val_metrics




def dq_train_mvnn(SATS_auction_instance,
                  capacity_generic_goods,
                  P_train,
                  X_train,
                  P_val,
                  X_val,
                  P_val_gen_only,
                  X_val_gen_only,
                  SATS_parameters,
                  TRAIN_parameters,
                  MVNN_parameters,
                  MIP_parameters,
                  bidder_id,
                  bidder_scale,
                  GSVM_national_bidder_goods_of_interest,
                  wandb_tracking
                  ):
    

    SATS_domain = SATS_parameters['SATS_domain']
  
    batch_size = TRAIN_parameters['batch_size']
    if batch_size != 1:
        raise NotImplementedError('batch_size != 1 is not implemented yet')
    epochs = TRAIN_parameters['epochs'] 
    l2_reg = TRAIN_parameters['l2_reg']
    learning_rate = TRAIN_parameters['learning_rate']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    print_frequency = TRAIN_parameters['print_frequency']

    induced_values = []
    for i in range(len(P_train)):
        induced_values.append(np.dot(P_train[i], X_train[i]))


    train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(P_train).float()
                                                                  )


    train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                              batch_size= batch_size,
                                                              shuffle=True)
    if P_val is not None and X_val is not None:
        val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                    torch.from_numpy(P_val).float()
                                                                    )


        val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                batch_size= batch_size,
                                                                shuffle=True) 
        
    else:
        val_loader_demand_queries = None

    if P_val_gen_only is not None and X_val_gen_only is not None:
        val_dataset_gen_only = torch.utils.data.TensorDataset(torch.from_numpy(X_val_gen_only).float(),
                                                                    torch.from_numpy(P_val_gen_only).float()
                                                                    )


        val_loader_gen_only = torch.utils.data.DataLoader(val_dataset_gen_only,
                                                                batch_size= batch_size,
                                                                shuffle=False) 
        
    else:
        val_loader_gen_only = None



    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max'] 
    lin_skip_connection = MVNN_parameters['lin_skip_connection'] 
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']   
    init_little_const = MVNN_parameters['init_little_const']

    print('Creating MVNN model with parameters:')
    print(f'num_hidden_layers: {num_hidden_layers}')
    print(f'num_hidden_units: {num_hidden_units}')
    print(f'regularisation: {l2_reg}')
    print(f'learning_rate: {learning_rate}')
    print(f'clip_grad_norm: {clip_grad_norm}')

    if SATS_domain in ['GSVM', 'LSVM']:
        model = MVNN(input_dim=X_train.shape[1],
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const
                    )
    elif SATS_domain in ['SRVM', 'MRVM']:
        model = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                    num_hidden_layers=num_hidden_layers,
                    num_hidden_units=num_hidden_units,
                    layer_type=layer_type,
                    target_max=target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const,
                    capacity_generic_goods=capacity_generic_goods
                    )
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')



    # make sure ts have no regularisation (the bigger t the more regular)
    l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
    no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
    for p in [*model.named_parameters()]:
        if 'ts' in p[0]:
            no_l2_reg_parameters['params'].append(p[1])
        else:
            l2_reg_parameters['params'].append(p[1])

    optimizer = torch.optim.Adam([l2_reg_parameters,no_l2_reg_parameters],
                                 lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           float(epochs))

    metrics = {}

    if wandb_tracking: 
        for loss_str in ['train_loss_dq_scaled', 'val_loss_dq_scaled', 'mean_regret_scaled', 
                         'val_r2_scaled', 'val_KT_scaled', 'val_MAE_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs")

    for epoch in range(epochs):
        train_loss_dq = dq_train_mvnn_helper(model,
                                            optimizer,
                                            use_gradient_clipping,
                                            clip_grad_norm,
                                            train_loader_demand_queries,
                                            SATS_domain = SATS_domain,
                                            bidder_id = bidder_id,
                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest,
                                            device=torch.device('cpu'),
                                            MIP_parameters = MIP_parameters
                                            )
        if val_loader_demand_queries is not None:
            val_metrics = dq_val_mvnn(trained_model = model,
                                     val_loader = val_loader_demand_queries,
                                     val_loader_gen_only = val_loader_gen_only,
                                     train_loader = train_loader_demand_queries,
                                     SATS_auction_instance = SATS_auction_instance,
                                     SATS_domain= SATS_domain,
                                     GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                     bidder_id = bidder_id,
                                     scale = bidder_scale,
                                     device=torch.device('cpu'),
                                     MIP_parameters = MIP_parameters
                                     )

        scheduler.step()
        if val_loader_demand_queries is not None:
            metrics[epoch] = val_metrics
        else: 
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_dq_loss"] = train_loss_dq

        if wandb_tracking:
            wandb.log({f"Bidder_{bidder_id}_train_loss_dq": train_loss_dq, 
                       f"Bidder_{bidder_id}_val_loss_dq": val_metrics["scaled_dq_loss"], 
                       f"Bidder_{bidder_id}_mean_regret": val_metrics["mean_regret"], 
                       f"Bidder_{bidder_id}_val_r2": val_metrics["r2"], 
                       f"Bidder_{bidder_id}_val_KT": val_metrics["kendall_tau"], 
                       f"Bidder_{bidder_id}_val_MAE": val_metrics["mae"],
                       "epochs": epoch})

        # TODO: remove later since we have W&B
        if epoch % print_frequency == 0:
            if val_loader_demand_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}, val_dq_loss:{val_metrics["scaled_dq_loss"]:.5f}, val_mean_regret:{val_metrics["mean_regret"]:.5f}, val_r2:{val_metrics["r2"]:.5f}, val_kendall_tau:{val_metrics["kendall_tau"]:.5f}, val_mae:{val_metrics["mae"]:.5f}')
            else: 
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}')
    
    return model, metrics



# def generic_to_naive_mapping(bundle):
#     """
#     A mapping from generic representation to naive representation
#     """
#     bundle_naive = [0 for i in range(len(mrvm.get_good_ids()))]
#     for i in range(len(bundle)):
#         license_mapping = mrvm_generic.good_to_licence[i]
#         items_requested = bundle[i]
#         for j in range(items_requested):
#             bundle_naive[license_mapping[j]] = 1
#     return bundle_naive
    

def extend_prices(generic_price_vectors, capacities_generic_goods, good_to_license_mapping): 
    """
    A simple function that extends a price vector from the generic form to the naive form. 
    """
    # set_trace()
    all_extended_prices = []
    item_number = capacities_generic_goods.sum() 
    print('EXTEND PRICES CALLED, total item number: ', item_number)
    for generic_price_vector in generic_price_vectors: 
        prices_extended = [0 for i in range(item_number)]
        for i in range(len(generic_price_vector)): 
            identical_items = good_to_license_mapping[i]
            for j in identical_items:
                prices_extended[j] = generic_price_vector[i]
        all_extended_prices.append(prices_extended)

    all_extended_prices = np.array(all_extended_prices)
    print('---> ALL EXTENDED PRICES SHAPE:', all_extended_prices.shape)
    return all_extended_prices


def extend_bundles(generic_bundles, capacities_generic_goods, good_to_license_mapping):
    """
    A simple function that extends a bundle from the generic form to the naive form. 
    """
    all_extended_bundles = []
    item_number = capacities_generic_goods.sum()
    for generic_bundle in generic_bundles:
        bundle_extended = [0 for i in range(item_number)]
        for i in range(len(generic_bundle)):
            license_mapping_current_good = good_to_license_mapping[i]
            items_requested = generic_bundle[i]
            for j in range(items_requested):
                bundle_extended[license_mapping_current_good[j]] = 1

        all_extended_bundles.append(bundle_extended)

    all_extended_bundles = np.array(all_extended_bundles)
    print('---> ALL EXTENDED BUNDLES SHAPE:', all_extended_bundles.shape)
    
    return all_extended_bundles


def dq_train_mvnn_parallel(bidder_id,# bidder_id must be first position for joblib.parallel!!
                           capacity_generic_goods,
                           elicited_dqs,
                           dqs_val_data,
                           scales,
                           SATS_parameters,
                           TRAIN_parameters,
                           MVNN_parameters,
                           MIP_parameters,
                           GSVM_national_bidder_goods_of_interest,
                           wandb_tracking,
                           num_cpu_per_job, 
                           good_to_license_mapping
                        ):
    
    # Preparation for "PARALLEL TRAINING"
    # --------------------------------------
    print('DEBUG: INSIDE PARALLEL TRAINING LOOP')
    train_start_time = time.time()

    torch.set_num_threads(num_cpu_per_job)

    bidder_name = f'Bidder_{bidder_id}'
    TRAIN_parameters = TRAIN_parameters[bidder_name]
    MVNN_parameters = MVNN_parameters[bidder_name]

    P_train = elicited_dqs[bidder_name][1] / scales[bidder_name]  # scale the data to the range [0,1]
    X_train = elicited_dqs[bidder_name][0]
    # set_trace()
    if not TRAIN_parameters.get('generic_approach', True):
        P_train = extend_prices(P_train, capacity_generic_goods, good_to_license_mapping)
        X_train = extend_bundles(X_train, capacity_generic_goods, good_to_license_mapping)
        print('DEBUG: extended prices and bundles!!!')
    else:
        print('DEBUG: DID NOT extend prices and bundles!!!')
        print('DEBUG: TRAIN PARAMETERS --> ', TRAIN_parameters)
    
    if dqs_val_data:
        P_val = dqs_val_data[bidder_name][1]
        X_val = dqs_val_data[bidder_name][0]
        if not TRAIN_parameters.get('generic_approach', True):
            P_val = extend_prices(P_val, capacity_generic_goods, good_to_license_mapping)
            X_val = extend_bundles(X_val, capacity_generic_goods, good_to_license_mapping)
    else:
        P_val = None
        X_val = None
    # --------------------------------------
    # raise NotImplementedError('Parallel training is not implemented yet')

    SATS_domain = SATS_parameters['SATS_domain']
  
    batch_size = TRAIN_parameters['batch_size']
    if batch_size != 1:
        raise NotImplementedError('batch_size != 1 is not implemented yet')
    epochs = TRAIN_parameters['epochs'] 
    l2_reg = TRAIN_parameters['l2_reg']
    learning_rate = TRAIN_parameters['learning_rate']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    print_frequency = TRAIN_parameters['print_frequency']

    induced_values = []
    for i in range(len(P_train)):
        induced_values.append(np.dot(P_train[i], X_train[i]))


    train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(P_train).float()
                                                                  )


    train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                              batch_size= batch_size,
                                                              shuffle=True)
    if P_val is not None and X_val is not None:
        val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                    torch.from_numpy(P_val).float()
                                                                    )


        val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                batch_size= batch_size,
                                                                shuffle=True)
    else:
        val_loader_demand_queries = None

    print('DEBUG: made dataloaders')

    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max'] # TODO: check
    lin_skip_connection = MVNN_parameters['lin_skip_connection']
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']   
    init_little_const = MVNN_parameters['init_little_const']

    print('Creating MVNN model with parameters:')
    print(f'num_hidden_layers: {num_hidden_layers}')
    print(f'num_hidden_units: {num_hidden_units}')
    print(f'regularisation: {l2_reg}')
    print(f'learning_rate: {learning_rate}')
    print(f'clip_grad_norm: {clip_grad_norm}')

    if SATS_domain in ['GSVM', 'LSVM']:
        non_generic_approach = False 
        model = MVNN(input_dim=X_train.shape[1],
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const
                    )
    elif SATS_domain in ['SRVM', 'MRVM']:
        if TRAIN_parameters.get('generic_approach', True):
            non_generic_approach = False
            model = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                        num_hidden_layers=num_hidden_layers,
                        num_hidden_units=num_hidden_units,
                        layer_type=layer_type,
                        target_max=target_max,
                        lin_skip_connection = lin_skip_connection,
                        dropout_prob = dropout_prob,
                        init_method = init_method,
                        random_ts = random_ts,
                        trainable_ts = trainable_ts,
                        init_E = init_E,
                        init_Var = init_Var,
                        init_b = init_b,
                        init_bias = init_bias,
                        init_little_const = init_little_const,
                        capacity_generic_goods=capacity_generic_goods
                        )
        else: 
            print('USING NON-GENERIC MVNN for SRVM/MRVM!!!!')
            non_generic_approach = True
            model = MVNN(input_dim=X_train.shape[1],
                    num_hidden_layers = num_hidden_layers,
                    num_hidden_units = num_hidden_units,
                    layer_type = layer_type,
                    target_max = target_max,
                    lin_skip_connection = lin_skip_connection,
                    dropout_prob = dropout_prob,
                    init_method = init_method,
                    random_ts = random_ts,
                    trainable_ts = trainable_ts,
                    init_E = init_E,
                    init_Var = init_Var,
                    init_b = init_b,
                    init_bias = init_bias,
                    init_little_const = init_little_const
                    )
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_domain}')



    # make sure ts have no regularisation (the bigger t the more regular)
    l2_reg_parameters = {'params': [], 'weight_decay': l2_reg}
    no_l2_reg_parameters = {'params': [], 'weight_decay': 0.0}
    for p in [*model.named_parameters()]:
        if 'ts' in p[0]:
            no_l2_reg_parameters['params'].append(p[1])
        else:
            l2_reg_parameters['params'].append(p[1])

    optimizer = torch.optim.Adam([l2_reg_parameters,no_l2_reg_parameters],
                                 lr = learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           float(epochs))

    metrics = {}

    if wandb_tracking: 
        for loss_str in ['train_loss_dq_scaled', 'val_loss_dq_scaled', 'mean_regret_scaled', 
                         'val_r2_scaled', 'val_KT_scaled', 'val_MAE_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs")

    for epoch in range(epochs):
        train_loss_dq = dq_train_mvnn_helper(model,
                                            optimizer,
                                            use_gradient_clipping,
                                            clip_grad_norm,
                                            train_loader_demand_queries,
                                            SATS_domain = SATS_domain,
                                            bidder_id = bidder_id,
                                            GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest,
                                            device=torch.device('cpu'),
                                            MIP_parameters = MIP_parameters, 
                                            non_generic_approach = non_generic_approach,
                                            )
        val_metrics = None
        if val_loader_demand_queries is not None:
            raise NotImplementedError('Parallel training does not support validation yet')
            # SATS_auction_instance not pickable for parallel training
            """ val_metrics = dq_val_mvnn(trained_model = model,
                                     val_loader = val_loader_demand_queries,
                                     SATS_auction_instance = SATS_auction_instance,
                                     SATS_domain= SATS_domain,
                                     GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                     bidder_id = bidder_id,
                                     scale = TRAIN_parameters['scales'][f"Bidder_{bidder_id}"],
                                     device=torch.device('cpu'),
                                     MIP_parameters = MIP_parameters
                                     ) """

        scheduler.step()
        if val_loader_demand_queries is not None:
            metrics[epoch] = val_metrics
        else: 
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_dq_loss"] = train_loss_dq

        if wandb_tracking:
            if val_loader_demand_queries is not None:
                wandb.log({f"Bidder_{bidder_id}_train_loss_dq_scaled": train_loss_dq, 
                        f"Bidder_{bidder_id}_val_loss_dq_scaled": val_metrics["scaled_dq_loss"], 
                        f"Bidder_{bidder_id}_mean_regret_scaled": val_metrics["scaled_mean_regret"], 
                        f"Bidder_{bidder_id}_val_r2_scaled": val_metrics["scaled_r2"], 
                        f"Bidder_{bidder_id}_val_KT_scaled": val_metrics["scaled_kendall_tau"], 
                        f"Bidder_{bidder_id}_val_MAE_scaled": val_metrics["scaled_mae"],
                        "epochs": epoch})
            else:
                wandb.log({f"Bidder_{bidder_id}_train_loss_dq_scaled": train_loss_dq,
                           "epochs": epoch})

        if epoch % print_frequency == 0:
            if val_loader_demand_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}, val_dq_loss:{val_metrics["scaled_dq_loss"]:.5f}, val_mean_regret:{val_metrics["mean_regret"]:.5f}, val_r2:{val_metrics["r2"]:.5f}, val_kendall_tau:{val_metrics["kendall_tau"]:.5f}, val_mae:{val_metrics["mae"]:.5f}')
            else: 
                print(f'Current epoch: {epoch:>4} | train_dq_loss:{train_loss_dq:.5f}')

    # NEW: for parallel training measure time here
    # --------------------------------------
    train_end_time = time.time()
    metrics["train_time_elapsed"] = train_end_time - train_start_time
    logging.info(f'Training time for {bidder_name}: {metrics["train_time_elapsed"]}')
    # --------------------------------------

    # New return format for parallel training
    # --------------------------------------
    return {bidder_name: [model, metrics]}


def vq_train_mvnn_parallel(bidder_id,  # bidder_id must be first position for joblib.parallel!!
                        capacity_generic_goods,
                        trained_models_and_metrics,
                        elicited_vqs,
                        vqs_val_data,
                        scales,
                        SATS_parameters,
                        TRAIN_parameters,
                        MVNN_parameters,
                        wandb_tracking,
                        num_cpu_per_job, 
                        good_to_license_mapping
                        ):
    
    # Preparation for "PARALLEL TRAINING"
    # --------------------------------------
    print('DEBUG: INSIDE PARALLEL TRAINING LOOP FOR VALUE QUERIES')
    train_start_time = time.time()

    torch.set_num_threads(num_cpu_per_job)

    bidder_name = f'Bidder_{bidder_id}'
    TRAIN_parameters = TRAIN_parameters[bidder_name]
    MVNN_parameters = MVNN_parameters[bidder_name]

    Y_train = elicited_vqs[bidder_name][1] / scales[bidder_name]  # scale the data to the range [0,1]
    X_train = elicited_vqs[bidder_name][0]

    if not TRAIN_parameters.get('generic_approach', True):
        Y_train = extend_prices(Y_train, capacity_generic_goods, good_to_license_mapping)
        X_train = extend_bundles(X_train, capacity_generic_goods, good_to_license_mapping)
        print('DEBUG: extended prices and bundles!!!')
    else:
        print('DEBUG: DID NOT extend prices and bundles!!!')
        print('DEBUG: TRAIN PARAMETERS --> ', TRAIN_parameters)
    
    if vqs_val_data:
        Y_val = vqs_val_data[bidder_name][1]
        X_val = vqs_val_data[bidder_name][0]
        if not TRAIN_parameters.get('generic_approach', True):
            Y_val = extend_prices(Y_val, capacity_generic_goods, good_to_license_mapping)
            X_val = extend_bundles(X_val, capacity_generic_goods, good_to_license_mapping)
    else:
        Y_val = None
        X_val = None
    # --------------------------------------
    # raise NotImplementedError('Parallel training is not implemented yet')

  
    batch_size = TRAIN_parameters['batch_size_cardinal']
    # if batch_size != 1:
    #     raise NotImplementedError('batch_size != 1 is not implemented yet')
    epochs = TRAIN_parameters['epochs_cardinal'] 
    l2_reg = TRAIN_parameters['l2_reg_cardinal']
    learning_rate = TRAIN_parameters['learning_rate_cardinal']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    loss_cardinal = TRAIN_parameters['loss_cardinal']
    print_frequency = TRAIN_parameters['print_frequency']



    train_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(Y_train).float()
                                                                  )

    train_loader_value_queries = torch.utils.data.DataLoader(train_dataset_value_queries,
                                                             batch_size= batch_size,
                                                             shuffle=True)
    if Y_val is not None and X_val is not None:
        val_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                   torch.from_numpy(Y_val).float()
                                                                   )

        val_loader_value_queries = torch.utils.data.DataLoader(val_dataset_value_queries,
                                                               batch_size= batch_size,
                                                               shuffle=True)
    else:
        val_loader_value_queries = None

    print('DEBUG: made dataloaders for value queries')

    model, metrics = trained_models_and_metrics[bidder_name]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    if wandb_tracking: 
        for loss_str in ['train_loss_vq_scaled', 'val_loss_vq_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs_cardinal")

    for epoch in range(epochs):
        train_loss_vq = vq_train_mvnn_helper(model,
                                             optimizer,
                                             use_gradient_clipping,
                                             clip_grad_norm,
                                             train_loader_value_queries,
                                             torch.device('cpu'),
                                             loss_type=loss_cardinal)

        val_metrics = None
        if val_loader_value_queries is not None:
            raise NotImplementedError('Parallel training does not support validation yet')

        scheduler.step()
        if val_loader_value_queries is not None:
            metrics[epoch] = val_metrics
        else: 
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_vq_loss"] = train_loss_vq

        if wandb_tracking:
            if val_loader_value_queries is not None:
                wandb.log({f"Bidder_{bidder_id}_train_loss_vq_scaled": train_loss_vq, 
                           f"Bidder_{bidder_id}_val_loss_vq_scaled": val_metrics["scaled_vq_loss"],
                           "epochs": epoch})
            else:
                wandb.log({f"Bidder_{bidder_id}_train_loss_vq_scaled": train_loss_vq,
                           "epochs": epoch})

        if epoch % print_frequency == 0:
            if val_loader_value_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_vq_loss:{train_loss_vq:.5f}, val_vq_loss:{val_metrics["scaled_vq_loss"]:.5f}')
            else: 
                print(f'Current epoch: {epoch:>4} | train_vq_loss:{train_loss_vq:.5f}')

    # NEW: for parallel training measure time here
    # --------------------------------------
    train_end_time = time.time()
    metrics["train_time_elapsed"] += (train_end_time - train_start_time)  # accumulate time
    logging.info(f'Training time for value queries for {bidder_name}: {metrics["train_time_elapsed"]}')
    # --------------------------------------

    # New return format for parallel training
    # --------------------------------------
    return {bidder_name: [model, metrics]}


def vq_train_mvnn(trained_models_and_metrics,
                  capacity_generic_goods,
                  elicited_vqs,
                  vqs_val_data,
                  scales,
                  SATS_parameters,
                  TRAIN_parameters,
                  MVNN_parameters,
                  bidder_id,
                  wandb_tracking,
                  good_to_license_mapping):
    """
    Takes as input a model trained on Demand Queries (i.e., after the first phase). 
    And finalizes its raining on Value Queries.
    """
    
    # Preparation for Training on Value Queries
    # --------------------------------------
    print('DEBUG: INSIDE TRAINING LOOP FOR VALUE QUERIES')
    train_start_time = time.time()

    bidder_name = f'Bidder_{bidder_id}'
    TRAIN_parameters = TRAIN_parameters[bidder_name]
    MVNN_parameters = MVNN_parameters[bidder_name]

    Y_train = elicited_vqs[bidder_name][1] / scales[bidder_name]  # scale the data to the range [0,1]
    X_train = elicited_vqs[bidder_name][0]

    if not TRAIN_parameters.get('generic_approach', True):
        Y_train = extend_prices(Y_train, capacity_generic_goods, good_to_license_mapping)
        X_train = extend_bundles(X_train, capacity_generic_goods, good_to_license_mapping)
        print('DEBUG: extended prices and bundles!!!')
    else:
        print('DEBUG: DID NOT extend prices and bundles!!!')
        print('DEBUG: TRAIN PARAMETERS --> ', TRAIN_parameters)
    
    if vqs_val_data:
        Y_val = vqs_val_data[bidder_name][1]
        X_val = vqs_val_data[bidder_name][0]
        if not TRAIN_parameters.get('generic_approach', True):
            Y_val = extend_prices(Y_val, capacity_generic_goods, good_to_license_mapping)
            X_val = extend_bundles(X_val, capacity_generic_goods, good_to_license_mapping)
    else:
        Y_val = None
        X_val = None
    # --------------------------------------
  
    batch_size = TRAIN_parameters['batch_size_cardinal']
    epochs = TRAIN_parameters['epochs_cardinal']
    l2_reg = TRAIN_parameters['l2_reg_cardinal']
    learning_rate = TRAIN_parameters['learning_rate_cardinal']
    clip_grad_norm = TRAIN_parameters['clip_grad_norm']
    use_gradient_clipping = TRAIN_parameters['use_gradient_clipping']
    loss_cardinal = TRAIN_parameters['loss_cardinal']
    print_frequency = TRAIN_parameters['print_frequency']

    train_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                  torch.from_numpy(Y_train).float())

    train_loader_value_queries = torch.utils.data.DataLoader(train_dataset_value_queries,
                                                             batch_size=batch_size,
                                                             shuffle=True)
    if Y_val is not None and X_val is not None:
        val_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                   torch.from_numpy(Y_val).float())

        val_loader_value_queries = torch.utils.data.DataLoader(val_dataset_value_queries,
                                                               batch_size=batch_size,
                                                               shuffle=True)
    else:
        val_loader_value_queries = None

    print('DEBUG: made dataloaders for value queries')

    model, metrics = trained_models_and_metrics[bidder_name]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    if wandb_tracking:
        for loss_str in ['train_loss_vq_scaled', 'val_loss_vq_scaled']:
            wandb.define_metric(f'Bidder_{bidder_id}_{loss_str}', step_metric="epochs_cardinal")

    for epoch in range(epochs):
        train_loss_vq = vq_train_mvnn_helper(model,
                                             optimizer,
                                             use_gradient_clipping,
                                             clip_grad_norm,
                                             train_loader_value_queries,
                                             torch.device('cpu'),
                                             loss_type=loss_cardinal)

        val_metrics = None
        if val_loader_value_queries is not None:
            val_metrics = dq_val_mvnn(trained_model=model,
                                      val_loader=val_loader_value_queries,
                                      train_loader=train_loader_value_queries,
                                      SATS_auction_instance=None,
                                      SATS_domain=SATS_parameters['SATS_domain'],
                                      GSVM_national_bidder_goods_of_interest=None,
                                      bidder_id=bidder_id,
                                      scale=scales[bidder_name],
                                      device=torch.device('cpu'),
                                      MIP_parameters=None)

        scheduler.step()
        if val_loader_value_queries is not None:
            metrics[epoch] = val_metrics
        else:
            metrics[epoch] = {}
        metrics[epoch]["train_scaled_vq_loss"] = train_loss_vq

        
        if wandb_tracking:
            if val_loader_value_queries is not None:
                wandb.log({f"Bidder_{bidder_id}_train_loss_vq_scaled": train_loss_vq,
                           f"Bidder_{bidder_id}_val_loss_vq_scaled": val_metrics["scaled_vq_loss"],
                           "epochs": epoch})
            else:
                wandb.log({f"Bidder_{bidder_id}_train_loss_vq_scaled": train_loss_vq,
                           "epochs": epoch})

        if epoch % print_frequency == 0:
            if val_loader_value_queries is not None:
                print(f'Current epoch: {epoch:>4} | train_vq_loss:{train_loss_vq:.5f}, val_vq_loss:{val_metrics["scaled_vq_loss"]:.5f}')
            else:
                print(f'Current epoch: {epoch:>4} | train_vq_loss:{train_loss_vq:.5f}')

    train_end_time = time.time()
    metrics["train_time_elapsed"] += (train_end_time - train_start_time)
    logging.info(f'Training time for value queries for {bidder_name}: {metrics["train_time_elapsed"]}')

    return model, metrics


def train_mvnn_parallel_mixed(bidder_id,  # bidder_id must be first position for joblib.parallel!!
                              capacity_generic_goods,
                              elicited_dqs,  # the elicited demand queries to train on during the auction  
                              dqs_val_data,   # validation demand queries (if available)
                              elicited_vqs,   # the elicited value queries to train on during the auction
                              vqs_val_data,  # validation value queries (if available)
                              scales,
                              SATS_parameters,
                              TRAIN_parameters,
                              MVNN_parameters,
                              MIP_parameters,
                              GSVM_national_bidder_goods_of_interest,
                              num_cpu_per_job,
                              good_to_license_mapping, 
                              SATS_auction_instance,  # only needed for validation 
                              val_frequency = 20,
                              NOMU_train = False,   # if true: will also train an UB-MVNN using the NOMU framework 
                              TRAIN_parameters_NOMU = None,
                              MVNN_parameters_NOMU = None, 
                              ):
    """
    A function that alternates between training on demand queries and value queries using two different optimizers.
    """
    
    # Preparation for "PARALLEL TRAINING"
    # --------------------------------------
    print('DEBUG: INSIDE PARALLEL TRAINING LOOP WITH MIXED QUERIES')
    train_start_time = time.time()

    torch.set_num_threads(num_cpu_per_job)

    bidder_name = f'Bidder_{bidder_id}'
    TRAIN_parameters = TRAIN_parameters[bidder_name]
    MVNN_parameters = MVNN_parameters[bidder_name]



    epochs = TRAIN_parameters['epochs']
    use_cached_solution_frequency = TRAIN_parameters['cached_solution_frequency'] # how often to use the cached solution for the MIPs

    if NOMU_train:
        TRAIN_parameters_NOMU = TRAIN_parameters_NOMU[bidder_name]
        MVNN_parameters_NOMU = MVNN_parameters_NOMU[bidder_name]
        epochs_NOMU = TRAIN_parameters_NOMU['epochs']
        use_cached_solution_frequency_NOMU = TRAIN_parameters_NOMU['cached_solution_frequency'] # how often to use the cached solution for the MIPs


    # Demand Queries Data Preparation
    P_train = elicited_dqs[bidder_name][1] / scales[bidder_name]  # scale the data to the range [0,1]
    X_train = elicited_dqs[bidder_name][0]
    if not TRAIN_parameters.get('generic_approach', True):
        P_train = extend_prices(P_train, capacity_generic_goods, good_to_license_mapping)
        X_train = extend_bundles(X_train, capacity_generic_goods, good_to_license_mapping)
        print('DEBUG: extended prices and bundles for demand queries!!!')
    else:
        print('DEBUG: DID NOT extend prices and bundles for demand queries!!!')
        print('DEBUG: TRAIN PARAMETERS --> ', TRAIN_parameters)
    
    if dqs_val_data:
        P_val = dqs_val_data[bidder_name][1] / scales[bidder_name]
        X_val = dqs_val_data[bidder_name][0]
        if not TRAIN_parameters.get('generic_approach', True):
            P_val = extend_prices(P_val, capacity_generic_goods, good_to_license_mapping)
            X_val = extend_bundles(X_val, capacity_generic_goods, good_to_license_mapping)
    else:
        P_val = None
        X_val = None

    if P_train is not None and len(P_train) > 0: 
        train_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                                    torch.from_numpy(P_train).float())

        train_loader_demand_queries = torch.utils.data.DataLoader(train_dataset_demand_queries,
                                                                batch_size=TRAIN_parameters['batch_size_dq'],
                                                                shuffle=True)
        
    else:
        train_loader_demand_queries = None
        print('DEBUG: NO DEMAND QUERIES TO TRAIN ON!!!')

    if P_val is not None and X_val is not None:
        val_dataset_demand_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(),
                                                                    torch.from_numpy(P_val).float())

        val_loader_demand_queries = torch.utils.data.DataLoader(val_dataset_demand_queries,
                                                                batch_size=TRAIN_parameters['batch_size_dq'],
                                                                shuffle=True)
    else:
        val_loader_demand_queries = None

    
    # --- Value Queries Data Preparation --- # 
    if elicited_vqs[bidder_name] is None or len(elicited_vqs[bidder_name][0]) == 0:
        train_loader_value_queries = None
    
    else:
        Y_train = np.array(elicited_vqs[bidder_name][1]) / scales[bidder_name]  # scale the data to the range [0,1]
        X_train_vq = np.array(elicited_vqs[bidder_name][0])


        if not TRAIN_parameters.get('generic_approach', True):
            Y_train = extend_prices(Y_train, capacity_generic_goods, good_to_license_mapping)
            X_train_vq = extend_bundles(X_train_vq, capacity_generic_goods, good_to_license_mapping)
            print('DEBUG: extended prices and bundles for value queries!!!')
        else:
            print('DEBUG: DID NOT extend bundles for value queries!!!')
            print('DEBUG: TRAIN PARAMETERS --> ', TRAIN_parameters)
        
    

        train_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_train_vq).float(),
                                                                    torch.from_numpy(Y_train).float())

        train_loader_value_queries = torch.utils.data.DataLoader(train_dataset_value_queries,
                                                                batch_size=TRAIN_parameters['batch_size_vq'],
                                                                shuffle=True)
        
    
    if vqs_val_data:
        Y_val = np.array(vqs_val_data[bidder_name][1]) / scales[bidder_name]
        X_val_vq = np.array(vqs_val_data[bidder_name][0])
        if not TRAIN_parameters.get('generic_approach', True):
            X_val_vq = extend_bundles(X_val_vq, capacity_generic_goods, good_to_license_mapping)
    
    else:
        Y_val = None
        X_val_vq = None

        
    if Y_val is not None and X_val_vq is not None:
        val_dataset_value_queries = torch.utils.data.TensorDataset(torch.from_numpy(X_val_vq).float(),
                                                                torch.from_numpy(Y_val).float())

        val_loader_value_queries = torch.utils.data.DataLoader(val_dataset_value_queries,
                                                                batch_size=TRAIN_parameters['batch_size_vq'],
                                                                shuffle=True)
    else:
        val_loader_value_queries = None

    print('DEBUG: made dataloaders for both demand and value queries')

    # Model and Optimizer Setup
    num_hidden_layers = MVNN_parameters['num_hidden_layers']
    num_hidden_units = MVNN_parameters['num_hidden_units']
    layer_type = MVNN_parameters['layer_type']
    target_max = MVNN_parameters['target_max']
    lin_skip_connection = MVNN_parameters['lin_skip_connection']
    dropout_prob = MVNN_parameters['dropout_prob']
    init_method = MVNN_parameters['init_method']
    random_ts = MVNN_parameters['random_ts']
    trainable_ts = MVNN_parameters['trainable_ts']
    init_E = MVNN_parameters['init_E']
    init_Var = MVNN_parameters['init_Var']
    init_b = MVNN_parameters['init_b']
    init_bias = MVNN_parameters['init_bias']
    init_little_const = MVNN_parameters['init_little_const']

    print('Creating MVNN model with parameters:')
    print(f'num_hidden_layers: {num_hidden_layers}')
    print(f'num_hidden_units: {num_hidden_units}')
    print(f'regularisation: {TRAIN_parameters["l2_reg"]}')
    print(f'learning rate: {TRAIN_parameters["learning_rate"]}')
    print(f'clip_grad_norm: {TRAIN_parameters["clip_grad_norm"]}')

    # also create the nomu model if need be
    if NOMU_train:
        num_hidden_layers_NOMU = MVNN_parameters_NOMU['num_hidden_layers']
        num_hidden_units_NOMU = MVNN_parameters_NOMU['num_hidden_units']
        layer_type_NOMU = MVNN_parameters_NOMU['layer_type']
        target_max_NOMU = MVNN_parameters_NOMU['target_max']
        lin_skip_connection_NOMU = MVNN_parameters_NOMU['lin_skip_connection']
        dropout_prob_NOMU = MVNN_parameters_NOMU['dropout_prob']
        init_method_NOMU = MVNN_parameters_NOMU['init_method']
        random_ts_NOMU = MVNN_parameters_NOMU['random_ts']
        trainable_ts_NOMU = MVNN_parameters_NOMU['trainable_ts']
        init_E_NOMU = MVNN_parameters_NOMU['init_E']
        init_Var_NOMU = MVNN_parameters_NOMU['init_Var']
        init_b_NOMU = MVNN_parameters_NOMU['init_b']
        init_bias_NOMU = MVNN_parameters_NOMU['init_bias']
        init_little_const_NOMU = MVNN_parameters_NOMU['init_little_const']

        print('Creating NOMU MVNN model with parameters:')
        print(f'num_hidden_layers: {num_hidden_layers_NOMU}')
        print(f'num_hidden_units: {num_hidden_units_NOMU}')
        print(f'regularisation: {TRAIN_parameters_NOMU["l2_reg"]}')
        print(f'learning rate: {TRAIN_parameters_NOMU["learning_rate"]}')
        print(f'clip_grad_norm: {TRAIN_parameters_NOMU["clip_grad_norm"]}')


    if SATS_parameters['SATS_domain'] in ['GSVM', 'LSVM']:
        non_generic_approach = False
        model = MVNN(input_dim=X_train.shape[1],
                     num_hidden_layers=num_hidden_layers,
                     num_hidden_units=num_hidden_units,
                     layer_type=layer_type,
                     target_max=target_max,
                     lin_skip_connection=lin_skip_connection,
                     dropout_prob=dropout_prob,
                     init_method=init_method,
                     random_ts=random_ts,
                     trainable_ts=trainable_ts,
                     init_E=init_E,
                     init_Var=init_Var,
                     init_b=init_b,
                     init_bias=init_bias,
                     init_little_const=init_little_const)
        
        if NOMU_train:
            UBmodel= MVNN(input_dim=X_train.shape[1],
                              num_hidden_layers=num_hidden_layers_NOMU,
                              num_hidden_units=num_hidden_units_NOMU,
                              layer_type=layer_type_NOMU,
                              target_max=target_max_NOMU,
                              lin_skip_connection=lin_skip_connection_NOMU,
                              dropout_prob=dropout_prob_NOMU,
                              init_method=init_method_NOMU,
                              random_ts=random_ts_NOMU,
                              trainable_ts=trainable_ts_NOMU,
                              init_E=init_E_NOMU,
                              init_Var=init_Var_NOMU,
                              init_b=init_b_NOMU,
                              init_bias=init_bias_NOMU,
                              init_little_const=init_little_const_NOMU)

    elif SATS_parameters['SATS_domain'] in ['SRVM', 'MRVM']:
        if TRAIN_parameters.get('generic_approach', True):
            non_generic_approach = False
            model = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                                 num_hidden_layers=num_hidden_layers,
                                 num_hidden_units=num_hidden_units,
                                 layer_type=layer_type,
                                 target_max=target_max,
                                 lin_skip_connection=lin_skip_connection,
                                 dropout_prob=dropout_prob,
                                 init_method=init_method,
                                 random_ts=random_ts,
                                 trainable_ts=trainable_ts,
                                 init_E=init_E,
                                 init_Var=init_Var,
                                 init_b=init_b,
                                 init_bias=init_bias,
                                 init_little_const=init_little_const,
                                 capacity_generic_goods=capacity_generic_goods)
            
            if NOMU_train:
                UBmodel = MVNN_GENERIC(input_dim=len(capacity_generic_goods),
                                       num_hidden_layers=num_hidden_layers_NOMU,
                                       num_hidden_units=num_hidden_units_NOMU,
                                       layer_type=layer_type_NOMU,
                                       target_max=target_max_NOMU,
                                       lin_skip_connection=lin_skip_connection_NOMU,
                                       dropout_prob=dropout_prob_NOMU,
                                       init_method=init_method_NOMU,
                                       random_ts=random_ts_NOMU,
                                       trainable_ts=trainable_ts_NOMU,
                                       init_E=init_E_NOMU,
                                       init_Var=init_Var_NOMU,
                                       init_b=init_b_NOMU,
                                       init_bias=init_bias_NOMU,
                                       init_little_const=init_little_const_NOMU,
                                       capacity_generic_goods=capacity_generic_goods)

        else:
            print('USING NON-GENERIC MVNN for SRVM/MRVM!!!!')
            non_generic_approach = True
            model = MVNN(input_dim=X_train.shape[1],
                         num_hidden_layers=num_hidden_layers,
                         num_hidden_units=num_hidden_units,
                         layer_type=layer_type,
                         target_max=target_max,
                         lin_skip_connection=lin_skip_connection,
                         dropout_prob=dropout_prob,
                         init_method=init_method,
                         random_ts=random_ts,
                         trainable_ts=trainable_ts,
                         init_E=init_E,
                         init_Var=init_Var,
                         init_b=init_b,
                         init_bias=init_bias,
                         init_little_const=init_little_const)
            
            if NOMU_train:
                UBmodel = MVNN(input_dim=X_train.shape[1],
                                num_hidden_layers=num_hidden_layers_NOMU,
                                num_hidden_units=num_hidden_units_NOMU,
                                layer_type=layer_type_NOMU,
                                target_max=target_max_NOMU,
                                lin_skip_connection=lin_skip_connection_NOMU,
                                dropout_prob=dropout_prob_NOMU,
                                init_method=init_method_NOMU,
                                random_ts=random_ts_NOMU,
                                trainable_ts=trainable_ts_NOMU,
                                init_E=init_E_NOMU,
                                init_Var=init_Var_NOMU,
                                init_b=init_b_NOMU,
                                init_bias=init_bias_NOMU,
                                init_little_const=init_little_const_NOMU)
    else:
        raise NotImplementedError(f'Unknown SATS domain: {SATS_parameters["SATS_domain"]}')

    # ---  Step 1: Train the mean models --- #
    # Create two optimizers with different learning rates
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_parameters['learning_rate'], weight_decay=TRAIN_parameters['l2_reg'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs))

    metrics = {}

    
    for epoch in range(epochs):
        if train_loader_demand_queries is not None:  # only train on DQs if they are available
            if epoch % use_cached_solution_frequency == 0:
                train_loss_dq, cached_dataloader_demand_queries  = dq_train_mvnn_helper(model = model,
                                                    optimizer= optimizer,
                                                    use_gradient_clipping= TRAIN_parameters['use_gradient_clipping'],
                                                    clip_grad_norm= TRAIN_parameters['clip_grad_norm'],
                                                    train_loader_demand_queries = train_loader_demand_queries,
                                                    SATS_domain= SATS_parameters['SATS_domain'],
                                                    bidder_id= bidder_id,
                                                    GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                                    device = torch.device('cpu'),
                                                    MIP_parameters= MIP_parameters,
                                                    non_generic_approach= non_generic_approach, 
                                                    return_cached_dataloader= True)

            else: 
                train_loss_dq = dq_train_mvnn_helper_cached_dataloader(model = model,
                                                        optimizer = optimizer,
                                                        use_gradient_clipping = TRAIN_parameters['use_gradient_clipping'],
                                                        clip_grad_norm = TRAIN_parameters['clip_grad_norm'],
                                                        train_loader_demand_queries_cached = cached_dataloader_demand_queries,
                                                        device = torch.device('cpu'))
                
        else:
            train_loss_dq = None

        if train_loader_value_queries is not None: # only train on VQs if they are available
            train_loss_vq, train_metrics_vqs = vq_train_mvnn_helper(model = model,
                                            optimizer= optimizer,
                                            use_gradient_clipping= TRAIN_parameters['use_gradient_clipping'],
                                            clip_grad_norm= TRAIN_parameters['clip_grad_norm'],
                                            train_loader_value_queries= train_loader_value_queries,
                                            device= torch.device('cpu'),
                                            loss_type= TRAIN_parameters['loss_vq'])
        else: 
            train_loss_vq = None
        
        scheduler.step()
        model.transform_weights()  # transform the weights to ensure it is a legal mvnn 

        val_metrics_demand_queries = None
        val_metrics_value_queries = None

        if val_loader_demand_queries is not None and ((epoch + 1) % val_frequency == 0 or epoch == epochs - 1):
            val_metrics_demand_queries = dq_val_mvnn(trained_model=model,
                                      val_loader=val_loader_demand_queries,
                                      train_loader=train_loader_demand_queries,
                                      SATS_auction_instance=SATS_auction_instance,
                                      SATS_domain=SATS_parameters['SATS_domain'],
                                      GSVM_national_bidder_goods_of_interest=GSVM_national_bidder_goods_of_interest,
                                      bidder_id=bidder_id,
                                      scale=scales[bidder_name],
                                      device=torch.device('cpu'),
                                      MIP_parameters=MIP_parameters)
            
        if val_loader_value_queries is not None and ((epoch + 1) % val_frequency == 0 or epoch == epochs - 1):
            val_metrics_value_queries = vq_val_mvnn(trained_model=model,
                                val_loader=val_loader_value_queries,
                                scale=scales[bidder_name],
                                device=torch.device('cpu'))

        metrics[epoch] = {}
        if val_metrics_demand_queries is not None:
            metrics[epoch]['val_metrics_demand_queries'] = val_metrics_demand_queries
        
        if val_metrics_value_queries is not None:
            metrics[epoch]['val_metrics_value_queries'] = val_metrics_value_queries

        if train_loss_dq is not None:
            metrics[epoch]["train_scaled_dq_loss"] = train_loss_dq

        if train_loss_vq is not None:
            metrics[epoch]["train_scaled_vq_loss"] = train_loss_vq
            metrics[epoch]["train_metrics_vqs"] = train_metrics_vqs

        if epoch % TRAIN_parameters['print_frequency'] == 0:
            string_to_print = f'Current epoch: {epoch:>4} | '
            if train_loss_dq is not None:
                string_to_print += f'train_dq_loss:{train_loss_dq:.5f}, '
            if train_loss_vq is not None:
                string_to_print += f'train_vq_loss:{train_loss_vq:.5f}, '
            if val_metrics_demand_queries is not None:
                string_to_print += f'val_dq_loss:{val_metrics_demand_queries["scaled_dq_loss"]:.5f}, '
            if val_metrics_value_queries is not None:
                string_to_print += f'val_vq_loss:{val_metrics_value_queries["mae"]:.5f}'

            print(string_to_print)
                  


    # Measure training time
    train_end_time = time.time()
    metrics["train_time_elapsed"] = train_end_time - train_start_time
    logging.info(f'Training time for {bidder_name}: {metrics["train_time_elapsed"]}')

    if not NOMU_train:
        return {bidder_name: [model, metrics]}
    
    
    # # --- Now that the mean model is trained, Train the UB-MVNN model --- #
    print('Training the UB-MVNN model')
    # set_trace()
    train_time_nomu_start = time.time()

    # Step 1: Create two optimizers with different learning rates for the UB-MVNN
    optimizer_NOMU = torch.optim.Adam(UBmodel.parameters(), lr=TRAIN_parameters_NOMU['learning_rate'], weight_decay=TRAIN_parameters_NOMU['l2_reg'])
    scheduler_NOMU = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_NOMU, float(epochs_NOMU))


    # -- Also create a new value query dataloader so that we have the right batch size --- #
    train_loader_value_queries_nomu = torch.utils.data.DataLoader(train_dataset_value_queries,
                                                                batch_size=TRAIN_parameters_NOMU['batch_size'],
                                                                shuffle=True)

    
    # Step 2: Create the 100% UB model (NOTE: uses scaled data, as everything else)
    ub_100_percent_model = Explicit100PercentUpperBoundIncludingDQS(X_vqs= X_train_vq,
                                                                    y_vqs= Y_train,
                                                                    X_dqs= X_train,
                                                                    P_dqs= P_train)
    

    # Step 3: Train the UB-MVNN model
    for epoch in range(epochs_NOMU):
        if epoch % use_cached_solution_frequency_NOMU == 0:
            train_loss_dq, cached_dataloader_demand_queries  = dq_train_mvnn_helper(model = UBmodel,
                                                 optimizer= optimizer_NOMU,
                                                 use_gradient_clipping= TRAIN_parameters_NOMU['use_gradient_clipping'],
                                                 clip_grad_norm= TRAIN_parameters_NOMU['clip_grad_norm'],
                                                 train_loader_demand_queries = train_loader_demand_queries,
                                                 SATS_domain= SATS_parameters['SATS_domain'],
                                                 bidder_id= bidder_id,
                                                 GSVM_national_bidder_goods_of_interest= GSVM_national_bidder_goods_of_interest,
                                                 device = torch.device('cpu'),
                                                 MIP_parameters= MIP_parameters,
                                                 non_generic_approach= non_generic_approach, 
                                                 return_cached_dataloader= True)

        else: 
            train_loss_dq = dq_train_mvnn_helper_cached_dataloader(model = UBmodel,
                                                    optimizer = optimizer_NOMU,
                                                    use_gradient_clipping = TRAIN_parameters_NOMU['use_gradient_clipping'],
                                                    clip_grad_norm = TRAIN_parameters_NOMU['clip_grad_norm'],
                                                    train_loader_demand_queries_cached = cached_dataloader_demand_queries,
                                                    device = torch.device('cpu'))




        
        NOMU_loss, NOMU_metrics =  nomu_hybrid_train_mvnn_helper(
                ub_model = UBmodel,
                mean_model = model,
                exp_upper_bound_net = ub_100_percent_model,
                NOMU_optimizer = optimizer_NOMU,
                use_gradient_clipping = TRAIN_parameters_NOMU['use_gradient_clipping'],
                clip_grad_norm = TRAIN_parameters_NOMU['clip_grad_norm'],
                train_loader_value_queries = train_loader_value_queries_nomu,
                device = torch.device('cpu'),
                NOMU_parameters = TRAIN_parameters_NOMU)

       
        
        scheduler_NOMU.step()

    train_nomu_end_time = time.time()
    print(f'Training time for NOMU model: {train_nomu_end_time - train_time_nomu_start}')
    # set_trace()

    
    return {bidder_name: [UBmodel, metrics]}


