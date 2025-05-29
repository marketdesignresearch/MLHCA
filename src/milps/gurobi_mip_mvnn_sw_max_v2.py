# Libs
import time
from timeit import default_timer as timer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
# from pdb import set_trace

#%% NEW SUCCINCT MVNN MIP FOR CALCULATING MAX UTILITY BUNDLE FOR SINGLE BIDDER: argmax_x {MVNN_i(x)-p*x}
class GUROBI_MIP_MVNN_MULTIPLE_BIDDERS_SW_MAX:

    def __init__(self,
                 models,  # list of all bider moddels 
                 scales,
                 capacities,
                 SATS_domain,
                 GSVM_national_bidder_goods_of_interest, 
                 forbid_single_bidder
                 ):
        
        # MVNN PARAMETERS
        self.models = models  # MVNN TORCH MODEL
        self.scales = scales  # SCALING FACTOR FOR EACH MODEL
        self.ts_all_models = [[layer.ts.data.cpu().detach().numpy().reshape(-1, 1) for layer in model.layers] for model in self.models]  # TS FOR ALL MODELS
        # MIP VARIABLES
        self.y_variables_all_models = [[] for _ in range(len(self.models))]  # CONTINUOUS VARS (y)
        self.a_variables_all_models = [[] for _ in range(len(self.models))]  # BINARY VARS 1
        self.b_variables_all_models = [[] for _ in range(len(self.models))]  # BINARY VARS 2
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}
        # SATS PARAMETERS
        self.SATS_domain = SATS_domain
        self.capacities = capacities
        self.GSVM_national_bidder_goods_of_interest = GSVM_national_bidder_goods_of_interest

        # parameters needed when solving the same MIP n-times for the n-bidders with different allocation constraints/marginalized bidders
        self.forbid_single_bidder = forbid_single_bidder
        self.allocations_to_forbid = []
        self.marginalized_bidder = - 1

    def calc_preactivated_box_bounds(self,
                                     input_upper_bound=1,
                                     input_lower_bound=0,
                                     verbose = False):

        # BOX-bounds for y variable (preactivated!!!!) as column vectors

        # Initialize
        self.upper_box_bounds_all_models = [[np.array([input_upper_bound] * model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)] for model in self.models]
        self.lower_box_bounds_all_models = [[np.array([input_lower_bound] * model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)] for model in self.models]

        # Propagate through Network
        for (model_number, model) in enumerate(self.models):
            for i, layer in enumerate(model.layers):
                W = layer.weight.data.cpu().detach().numpy()
                b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)

                # -------------------
                if i == 0:
                    # Remark: no vector t needed since we calculate pre-activate bounds for first hidden layer, thus no self.phi() here
                    self.upper_box_bounds_all_models[model_number].append(W @ self.upper_box_bounds_all_models[model_number][-1] + b)
                    self.lower_box_bounds_all_models[model_number].append(W @ self.lower_box_bounds_all_models[model_number][-1] + b)
                else:
                    # Remark: no we need for preactivated bounds of the ith hidden layer the ts from the previous layer i.e. i-1
                    t = self.ts_all_models[model_number][i-1]
                    self.upper_box_bounds_all_models[model_number].append(W @ self.phi(self.upper_box_bounds_all_models[model_number][-1], t) + b)
                    self.lower_box_bounds_all_models[model_number].append(W @ self.phi(self.lower_box_bounds_all_models[model_number][-1], t) + b)
                # -------------------

        if verbose:
            print('Upper Box Bounds:')
            print(self.upper_box_bounds_all_models)
        if verbose:
            print('Lower Box Bounds:')
            print(self.lower_box_bounds_all_models)

    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)

    def generate_mip(self,
                     MIPGap = None,
                     verbose = False,
                     ):

        self.mip = gp.Model("MVNN MIP2")

        # Add IntFeasTol, primal feasibility
        if MIPGap:
            self.mip.Params.MIPGap = MIPGap

        self.calc_preactivated_box_bounds(verbose=verbose)

        # --- Variable declaration -----
        for model_number, model in enumerate(self.models):
            
            self.y_variables_all_models[model_number].append(self.mip.addVars([i for i in range(len(self.capacities))], name=f'x_{model_number}_', vtype = GRB.BINARY))  # the "input variables, i.e. the first y level"

            for (i, layer) in enumerate(model.layers):

                # ----------------------------
                tmp_y_variables = []
                for j in range(len(layer.weight.data)):
                    tmp_y_variables.append(self.mip.addVar(name=f'y_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub = self.ts_all_models[model_number][i][j, 0]))
                self.y_variables_all_models[model_number].append(tmp_y_variables)
                # ----------------------------

                self.a_variables_all_models[model_number].append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'a_{model_number}_{i+1}_', vtype = GRB.BINARY))
                self.b_variables_all_models[model_number].append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'b_{model_number}_{i+1}_', vtype = GRB.BINARY))

            layer = model.output_layer

            self.y_variables_all_models[model_number].append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'y_output_{model_number}_', vtype = GRB.CONTINUOUS, lb = 0))

            # ---  MVNN Contraints ---
            for (i, layer) in enumerate(model.layers):
                for (j, weight) in enumerate(layer.weight.data):
                    # CASE 1 -> REMOVAL:
                    # set_trace()
                    # print(i,j)
                    if self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= self.ts_all_models[model_number][i][j, 0]:
                        self.y_variables_all_models[model_number][i+1][j] = self.ts_all_models[model_number][i][j, 0]
                        self.case_counter['Case1'] += 1
                    # CASE 2 -> REMOVAL:
                    elif self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= 0:
                        self.y_variables_all_models[model_number][i+1][j] = 0
                        self.case_counter['Case2'] += 1
                    # CASE 3 -> REMOVAL:
                    elif (self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= 0 and self.lower_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i][j, 0]) and (self.upper_box_bounds_all_models[model_number][i+1][j, 0] >= 0 and self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i][j, 0]):
                        self.y_variables_all_models[model_number][i+1][j] = gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j]
                        self.case_counter['Case3'] += 1
                    # CASE 4 -> REMOVAL:
                    elif self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= 0:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.ts_all_models[model_number][i][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_number}+{i+1}_{j}_Case4_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= self.b_variables_all_models[model_number][i][j] * self.ts_all_models[model_number][i][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts_all_models[model_number][i][j, 0] - self.upper_box_bounds_all_models[model_number][i+1][j, 0]) * self.b_variables_all_models[model_number][i][j], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT4')
                        self.case_counter['Case4'] += 1
                    # CASE 5 -> REMOVAL:
                    elif self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i][j, 0]:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.a_variables_all_models[model_number][i][j] * self.ts_all_models[model_number][i][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_all_models[model_number][i+1][j, 0]*(1-self.a_variables_all_models[model_number][i][j]), name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= 0, name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT4')
                        self.case_counter['Case5'] += 1
                    # DEFAULT CASE -> NO REMOVAL:
                    else:
                        # TYPE 1 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.a_variables_all_models[model_number][i][j] * self.ts_all_models[model_number][i][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT1')
                        # TYPE 2 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_all_models[model_number][i+1][j, 0]*(1-self.a_variables_all_models[model_number][i][j]), name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT2')
                        # TYPE 3 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= self.b_variables_all_models[model_number][i][j] * self.ts_all_models[model_number][i][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT3')
                        # TYPE 4 Constraints for the whole network (except the output layer)
                        self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts_all_models[model_number][i][j, 0] - self.upper_box_bounds_all_models[model_number][i+1][j, 0]) * self.b_variables_all_models[model_number][i][j], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT4')

            output_weight = model.output_layer.weight.data[0]
            if (model.output_layer.bias is not None):
                output_bias = model.output_layer.bias.data
            else:
                output_bias = 0

            if output_bias!=0:
                raise ValueError('output_bias is not 0')

            # Final output layer of MVNN
            # Linear Constraints for the output layer WITH lin_skip_layer: W*y
            if hasattr(model, 'lin_skip_layer'):
                lin_skip_W = model.lin_skip_layer.weight.detach().cpu().numpy() 
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables_all_models[model_number][-2][k] for k in range(len(output_weight))) + output_bias + gp.quicksum(lin_skip_W[0, i]*self.y_variables_all_models[model_number][0][i] for i in range(lin_skip_W.shape[1])) == self.y_variables_all_models[model_number][-1][0], name=f'output_layer_model_{model_number}')
            # Linear Constraints for the output layer WIHTOUT lin_skip_layer: W*y + W_0*x
            else:
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables_all_models[model_number][-2][k] for k in range(len(output_weight))) + output_bias == self.y_variables_all_models[model_number][-1][0], name=f'output_layer_model_{model_number}')

        # GSVM specific allocation constraints
        if self.SATS_domain == 'GSVM':
            for model_number in range(len(self.models)):

                if model_number == 6:
                    #print(f'Adding GSVM specific constraints for national bidder: {self.bidder_id}')
                    GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement = [i not in self.GSVM_national_bidder_goods_of_interest for i in range(len(self.capacities))]
                    self.mip.addConstr(gp.quicksum(self.y_variables_all_models[model_number][0][i]*GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement[i] for i in range(len(self.capacities)))==0, name="GSVM_CT_NationalBidder")
                else:
                    #print(f'Adding GSVM specific constraints for regional bidder: {self.bidder_id}')
                    self.mip.addConstr(gp.quicksum(self.y_variables_all_models[model_number][0][i] for i in range(len(self.capacities)))<=4, name="GSVM_CT_RegionalBidder")
        

         # --- Capacity Constraints --- # 
        for item_index in range(len(self.capacities)):
            self.mip.addConstr(gp.quicksum(self.y_variables_all_models[bidder_index][0][item_index] for bidder_index in range(len(self.models))) <= self.capacities[item_index], name=f'item_capacity_{item_index}')

        
        # --- Objective Declaration ---
        # self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices[i] for i in range(len(prices))), GRB.MAXIMIZE)
        self.mip.setObjective(gp.quicksum(self.scales[model_number] * self.y_variables_all_models[model_number][-1][0] for model_number in range(len(self.models))), GRB.MAXIMIZE)

        if (verbose):
            self.mip.write('MVNN_mip2_'+'_'.join(time.ctime().replace(':', '-').split(' '))+'.lp')


    def add_forbidden_allocation(self, forbidden_allocation, forbid_individual_bundles = True):
        """
        Adds a constraint to the MIP model to exclude a specific allocation.

        :param forbidden_allocation: A list of lists where each sublist corresponds to the allocation of a bidder.
                                    Each sublist contains 1s and 0s indicating the presence or absence of items.
        """
        print('add forbidden allocation called')
        print('forbidden allocation:', forbidden_allocation)
        if len(forbidden_allocation) != len(self.models):
            raise ValueError("The forbidden allocation must have the same number of sublists as there are models.")
        
        # if any(len(allocation) != len(self.capacities) for allocation in forbidden_allocation):
        #     raise ValueError("Each sublist in the forbidden allocation must have the same length as the item capacities list.")

        if self.forbid_single_bidder:
            # If we forbid a single bidder each time -> 
            # Store all allocations we need to forbid, and add the right part of the constraints when solving the MIP n-times (for each bidder)
            self.allocations_to_forbid.append(forbidden_allocation)
            return 
        
        expr_list = []
        for model_index, allocation in enumerate(forbidden_allocation):
            expr_list_current_model = []
            for item_index, item in enumerate(allocation):
                if item == 1:
                    expr_list.append(self.y_variables_all_models[model_index][0][item_index])
                    expr_list_current_model.append(self.y_variables_all_models[model_index][0][item_index])
                else:
                    expr_list.append(1 - self.y_variables_all_models[model_index][0][item_index])
                    expr_list_current_model.append(1 - self.y_variables_all_models[model_index][0][item_index])
            
            if forbid_individual_bundles:
                self.mip.addConstr(gp.quicksum(expr_list_current_model) <= len(expr_list_current_model) - 1, name=f'alreadyQueried_bidder_{model_index}')

        if not forbid_individual_bundles:
            self.mip.addConstr(gp.quicksum(expr_list) <= len(expr_list) - 1, name='forbidden_allocation')
        self.mip.update()


    # def add_forbidden_bundles(self, forbidden_allocation):
    #     """
    #     Adds a constraint to the MIP model to exclude a specific allocation. 
    #     Excludes the allocation on a per-bidder basis, not as a hole. 

    #     :param forbidden_allocation: A list of lists where each sublist corresponds to the allocation of a bidder.
    #                                 Each sublist contains 1s and 0s indicating the presence or absence of items.
    #     """
         
    #     for model_index, allocation in enumerate(forbidden_allocation):
    #         expr_list = []
    #         for item_index, item in enumerate(allocation):
    #             if item == 1:
    #                 expr_list.append(self.y_variables_all_models[model_index][0][item_index])
    #             else:
    #                 expr_list.append(1 - self.y_variables_all_models[model_index][0][item_index])

    #         self.mip.addConstr(gp.quicksum(expr_list) <= len(expr_list) - 1, name=f'forbidden_bundle_bidder_{model_index}')
    #         self.mip.update()


    def marginalize_bidder(self, bidder_index):
        self.marginalized_bidder = bidder_index

        # get all already queried bundle constraints for that bidder, and remove them from the mip 
        # get all constraints that contain the bidder_index
        constraints_to_remove = [constr for constr in self.mip.getConstrs() if f'alreadyQueried_bidder_{bidder_index}' in constr.getAttr('ConstrName')]
        # remove them
        for constr in constraints_to_remove:
            self.mip.remove(constr)
        
        self.mip.update()

        # add the constraint that the specific bidder gets the zero bundle 
        # get all variables of the 0 level of that bidder. We cannot use self.y_model_all_models[bidder_index][0] because we have a copy of the original mip (and variables)
        y_vars = [var for var in self.mip.getVars() if f'x_{bidder_index}_' in var.getAttr('VarName')]
        expr = gp.quicksum(y_vars)
        self.mip.addConstr(expr == 0, name=f'bidder_{bidder_index}_marginalized') 
        self.mip.update()


    def forbid_bundles_single_bidder(self, bundles_to_forbid, model_index):
        """
        Takes as input a list of bundles and forbids those bundles for the specific bidder. 
        Also removes all constraints of the same type that refer to other bidders 
        """

        if model_index == self.marginalized_bidder:
            print('Bidder is marginalized. We do not forbid bundles.')
            return 

        # get all constraints that contain any model index 
        constraints_to_remove = [constr for constr in self.mip.getConstrs() if f'alreadyQueried_bidder_' in constr.getAttr('ConstrName')]
        # remove them
        print(f'Removing {len(constraints_to_remove)} constraints')
        for constr in constraints_to_remove:
            self.mip.remove(constr)
        
        self.mip.update()

        for bundle in bundles_to_forbid:
            expr_list = []
            for item_index, item in enumerate(bundle):
                if item == 1:
                    expr_list.append(self.y_variables_all_models[model_index][0][item_index])
                else:
                    expr_list.append(1 - self.y_variables_all_models[model_index][0][item_index])

            self.mip.addConstr(gp.quicksum(expr_list) <= len(expr_list) - 1, name=f'alreadyQueried_bidder_{model_index}')

        self.mip.update()


    def solve_mip(self,
                  outputFlag = False,
                  verbose = True,
                  timeLimit = np.inf,
                  MIPGap = 1e-04,
                  IntFeasTol = 1e-5,
                  FeasibilityTol = 1e-6, 
                  return_rv = False 
                  ):
        
        if not verbose:
            self.mip.Params.LogToConsole = 0
            self.mip.Params.OutputFlag = 0

        # set solve parameter (if not sepcified, default values are used)
        if not self.forbid_bundles_single_bidder:
            self.mip.Params.timeLimit = timeLimit # Default +inf
        else:
            self.mip.Params.timeLimit = (timeLimit * 2) / len(self.models) # Default +inf
        self.mip.Params.MIPGap = MIPGap # Default 1e-04
        self.mip.Params.IntFeasTol = IntFeasTol # Default 1e-5
        self.mip.Params.FeasibilityTol = FeasibilityTol # Default 1e-6
        

        self.start = timer()
        self.mip.Params.OutputFlag = outputFlag
        self.optimal_bundles = []
        
        if not self.forbid_single_bidder:
            self.mip.optimize()
            self.end = timer()

            
            try:
                for model_index in range(len(self.models)):
                    bundle = []
                    for i in range(len(self.capacities)):
                        if self.y_variables_all_models[model_index][0][i].x >= 0.99:
                            bundle.append(1)
                        else:
                            bundle.append(0)
                    self.optimal_bundles.append(bundle)
            except:
                self._print_info()
                raise ValueError('MIP did not solve successfully!')
            
        else:
            try: 
                self.start = timer()
                for model_index in range(len(self.models)):
                    print(f'Solving individual MIP for bidder {model_index}')

                    bundles_to_forbid_single_bidder = [allocation[model_index] for allocation in self.allocations_to_forbid] # find all bundles to forbid for that bidder 
                    self.forbid_bundles_single_bidder(bundles_to_forbid_single_bidder, model_index)
                    
                    
                    self.mip.optimize()
                    
                    bundle = []
                    for i in range(len(self.capacities)):
                        bundle.append(np.int64(np.round(self.y_variables_all_models[model_index][0][i].x)))
                    self.optimal_bundles.append(bundle)

                self.end = timer()

            except:
                # If an exception occurs during model optimization, generate IIS
                self.mip.computeIIS()
                # Write the IIS to a file
                self.mip.write('infeasible_model.ilp')
                    
                # self._print_info()
                raise ValueError('MIP did not solve successfully!')

        self.optimal_bundles = np.array(self.optimal_bundles)
        if verbose:
            self._print_info()

        if not return_rv:
            return self.optimal_bundles
        else:
            return self.optimal_bundles, self.mip.getObjective().getValue()


    def _print_info(self):
        print(*['*']*30)
        print('MIP INFO:')
        print(*['-']*30)
        print(f'Name: {self.mip.ModelName}')
        print(f'Goal: {self._model_sense_converter(self.mip.ModelSense)}')
        print(f'Objective: {self.mip.getObjective()}')
        print(f'Number of variables: {self.mip.NumVars}')
        print(f' - Binary {self.mip.NumBinVars}')
        print(f'Number of linear constraints: {self.mip.NumConstrs}')
        print(f'Primal feasibility tolerance for constraints: {self.mip.Params.FeasibilityTol}')
        print(f'Integer feasibility tolerance: {self.mip.Params.IntFeasTol}')
        print(f'Relative optimality gap: {self.mip.Params.MIPGap}')  # we may want this 
        print(f'Time Limit: {self.mip.Params.TimeLimit}')
        print('')
        print('MIP SOLUTION:')
        print(*['-']*30)
        print(f'Status: {self._status_converter(self.mip.status)}')
        print(f'Elapsed in sec: {self.end - self.start}')
        print(f'Reached Relative optimality gap: {self.mip.MIPGap}')   
        print(f'Optimal Allocation: {self.optimal_bundles}')
        print(f'Objective Value: {self.mip.ObjVal}')
        print(f'Number of stored solutions: {self.mip.SolCount}')
        print('IA Case Statistics:')
        for k, v in self.case_counter.items():
            print(f' - {k}: {v}')
        print(*['*']*30)

    def _status_converter(self, int_status):
        status_table = ['woopsies!', 'LOADED', 'OPTIMAL', 'INFEASIBLE', 'INF_OR_UNBD', 'UNBOUNDED', 'CUTOFF', 'ITERATION_LIMIT', 'NODE_LIMIT', 'TIME_LIMIT', 'SOLUTION_LIMIT', 'INTERRUPTED', 'NUMERIC', 'SUBOPTIMAL', 'INPROGRESS', 'USER_OBJ_LIMIT']
        return status_table[int_status]

    def _model_sense_converter(self, int_sense):
        if int_sense == 1:
            return 'Minimize'
        elif int_sense == -1:
            return 'Maximize'
        else:
            raise ValueError('int_sense needs to be -1:maximize or 1: minimize')
