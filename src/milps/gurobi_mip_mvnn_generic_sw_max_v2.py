# Libs
import time
from timeit import default_timer as timer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
# from pdb import set_trace

#%% NEW SUCCINCT MVNN MIP FOR CALCULATING MAX UTILITY BUNDLE FOR SINGLE BIDDER: argmax_x {MVNN_i(x)-p*x}
class GUROBI_MIP_MVNN_MULTIPLE_BIDDERS_GENERIC_SW_MAX:
    def __init__(self,
                 models,
                 capacities, 
                 scales,
                 forbid_single_bidder = False,
                 ):
        
        # Domain parameters
        self.capacities = capacities
        self.scales = scales
        
        
        # parameters needed when solving the same MIP n-times for the n-bidders with different allocation constraints/marginalized bidders
        self.forbid_single_bidder = forbid_single_bidder
        self.allocations_to_forbid = []
        self.marginalized_bidder = - 1
        
        # MVNN PARAMETERS
        self.models = models  # MVNN TORCH MODEL
        self.ts_all_models = [[layer.ts.data.cpu().detach().numpy().reshape(-1, 1) for layer in model.layers if layer._get_name() not in ['Linear']] for model in self.models]  # TS FOR ALL MODELS
        # MIP VARIABLES
        self.y_variables_all_models = [[] for _ in range(len(self.models))]  # CONTINUOUS VARS (y)
        self.a_variables_all_models = [[] for _ in range(len(self.models))]  # BINARY VARS 1
        self.b_variables_all_models = [[] for _ in range(len(self.models))]  # BINARY VARS 2
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}


        self.diff_variables_all_models = [[] for _ in range(len(self.models))]  # HELPER CONT VARS (for absolute value in "already-queried-bundle"-constraints)
        self.abs_variables_all_models = [[] for _ in range(len(self.models))]  # HELPER POSITVE CONT VARS (for absolute value in "already-queried-bundle"-constraints)
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}
 

    def calc_preactivated_box_bounds(self,
                                     verbose = False):

        # BOX-bounds for y variable (preactivated!!!!) as column vectors

        # Initialize
        input_upper_bounds = np.array(self.models[0].capacity_generic_goods, dtype=np.int64).reshape(-1, 1)
        
        self.upper_box_bounds_all_models = [[input_upper_bounds] for _ in range(len(self.models))]
        self.lower_box_bounds_all_models = [[np.array([0] * self.models[0].layers[0].in_features, dtype=np.int64).reshape(-1, 1)] for _ in range(len(self.models))]

        
        # Propagate through Network 
        for (model_number, model) in enumerate(self.models):
            for i, layer in enumerate(model.layers):

                # -------------------
                if i == 0:
                    W = layer.weight.data.cpu().detach().numpy()
                    # Remark: no bias b
                    # Remark: no vector t since we calculate pre-activate bounds for first hidden layer (generic trafo)
                    self.upper_box_bounds_all_models[model_number].append(W @ self.upper_box_bounds_all_models[model_number][-1])
                    self.lower_box_bounds_all_models[model_number].append(W @ self.lower_box_bounds_all_models[model_number][-1])

                elif i == 1:
                    W = layer.weight.data.cpu().detach().numpy()
                    b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)
                    # Remark: no vector t needed since we calculate pre-activate bounds for second hidden layer (first hidden MVNN layer), thus no self.phi() here
                    self.upper_box_bounds_all_models[model_number].append(W @ self.upper_box_bounds_all_models[model_number][-1] + b)
                    self.lower_box_bounds_all_models[model_number].append(W @ self.lower_box_bounds_all_models[model_number][-1] + b)

                else:
                    W = layer.weight.data.cpu().detach().numpy()
                    b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)
                    # Remark: now we need for preactivated bounds of the ith hidden layer the ts from the previous layer i.e. i-1
                    # However, since self.ts has one dimension less than self.model.layers we need to access now (i-2)nd position which has the vector of t for the (i-1)st hidden layer
                    t = self.ts_all_models[model_number][i-2]
                    self.upper_box_bounds_all_models[model_number].append(W @ self.phi(self.upper_box_bounds_all_models[model_number][-1], t) + b)
                    self.lower_box_bounds_all_models[model_number].append(W @ self.phi(self.lower_box_bounds_all_models[model_number][-1], t) + b)
                # -------------------

        if verbose:
            print('Upper Box Bounds:')
            print(self.upper_box_bounds_all_models)
            print('Lower Box Bounds:')
            print(self.lower_box_bounds_all_models)
        return


    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)


    def generate_mip(self,
                     MIPGap = None,
                     verbose = False,
                     ):

        self.mip = gp.Model("MVNN GENERIC MIP")

        # Add IntFeasTol, primal feasibility
        if MIPGap:
            self.mip.Params.MIPGap = MIPGap

        self.calc_preactivated_box_bounds(verbose=verbose)

        for model_number, model in enumerate(self.models):
            # --- Variable declaration -----
            input_ubs = {i: self.upper_box_bounds_all_models[model_number][0][i, 0] for i in range(len(self.capacities))}
            input_lbs = {i: self.lower_box_bounds_all_models[model_number][0][i, 0] for i in range(len(self.capacities))}
            self.y_variables_all_models[model_number].append(self.mip.addVars([i for i in range(len(self.capacities))], name=f'x_{model_number}_', vtype = GRB.INTEGER, lb=input_lbs, ub=input_ubs))  # the "input variables, i.e. the first y level"

            for (i, layer) in enumerate(model.layers):
                tmp_y_variables = []
                tmp_a_variables = []
                tmp_b_variables = []
                # ----------------------------
                for j in range(len(layer.weight.data)):
                    if i == 0:
                        # NEW: first hidden layer after generic transformation has no cutoff t and an upper bound of 1
                        # ----------------
                        tmp_y_variables.append(self.mip.addVar(name=f'y_{model_number}_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub = 1))
                        # ----------------
                        # Remark no binary variables for first hidden layer after generic transformation
                    else:
                        tmp_y_variables.append(self.mip.addVar(name=f'y_{model_number}_{i+1}_{j}', vtype = GRB.CONTINUOUS, lb = 0, ub = self.ts_all_models[model_number][i-1][j, 0]))
                        tmp_a_variables.append(self.mip.addVar(name=f'a_{model_number}_{i+1}_{j}', vtype = GRB.BINARY))
                        tmp_b_variables.append(self.mip.addVar( name=f'b_{model_number}_{i+1}_{j}', vtype = GRB.BINARY))
                # ----------------------------
                self.y_variables_all_models[model_number].append(tmp_y_variables)
                if len(tmp_a_variables) > 0:
                    self.a_variables_all_models[model_number].append(tmp_a_variables)
                if len(tmp_b_variables) > 0:
                    self.b_variables_all_models[model_number].append(tmp_b_variables)

            layer = model.output_layer
            self.y_variables_all_models[model_number].append([self.mip.addVar(name=f'y_output_{model_number}_', vtype = GRB.CONTINUOUS, lb = 0)])

            # ---  MVNN Contraints ---
            # Remark: now we need to acces for self.y_variables[i+1] the self.ts[i-1], self.a_variables[i-1] and self.b_variables[i-1] !!!
            for (i, layer) in enumerate(model.layers):
                if i == 0:
                    # NEW: first hidden layer after generic transformation
                    for (j, weight) in enumerate(layer.weight.data):
                        self.y_variables_all_models[model_number][i+1][j] = gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) # no bias in generic transformation
                else:
                    for (j, weight) in enumerate(layer.weight.data):
                        # CASE 1 -> REMOVAL:
                        if self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= self.ts_all_models[model_number][i-1][j, 0]:
                            self.y_variables_all_models[model_number][i+1][j] = self.ts_all_models[model_number][i-1][j, 0]
                            self.case_counter['Case1'] += 1
                        # CASE 2 -> REMOVAL:
                        elif self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= 0:
                            self.y_variables_all_models[model_number][i+1][j] = 0
                            self.case_counter['Case2'] += 1
                        # CASE 3 -> REMOVAL:
                        elif (self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= 0 and self.lower_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i-1][j, 0]) and (self.upper_box_bounds_all_models[model_number][i+1][j, 0] >= 0 and self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i-1][j, 0]):
                            self.y_variables_all_models[model_number][i+1][j] = gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j]
                            self.case_counter['Case3'] += 1
                        # CASE 4 -> REMOVAL:
                        elif self.lower_box_bounds_all_models[model_number][i+1][j, 0] >= 0:
                            # TYPE 1 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.ts_all_models[model_number][i-1][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT1')
                            # TYPE 2 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT2')
                            # TYPE 3 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= self.b_variables_all_models[model_number][i-1][j] * self.ts_all_models[model_number][i-1][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT3')
                            # TYPE 4 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts_all_models[model_number][i-1][j, 0] - self.upper_box_bounds_all_models[model_number][i+1][j, 0]) * self.b_variables_all_models[model_number][i-1][j], name=f'HLayer_{model_number}_{i+1}_{j}_Case4_CT4')
                            self.case_counter['Case4'] += 1
                        # CASE 5 -> REMOVAL:
                        elif self.upper_box_bounds_all_models[model_number][i+1][j, 0] <= self.ts_all_models[model_number][i-1][j, 0]:
                            # TYPE 1 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.a_variables_all_models[model_number][i-1][j] * self.ts_all_models[model_number][i-1][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT1')
                            # TYPE 2 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_all_models[model_number][i+1][j, 0]*(1-self.a_variables_all_models[model_number][i-1][j]), name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT2')
                            # TYPE 3 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= 0, name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT3')
                            # TYPE 4 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_number}_{i+1}_{j}_Case5_CT4')
                            self.case_counter['Case5'] += 1
                        # DEFAULT CASE -> NO REMOVAL:
                        else:
                            # TYPE 1 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= self.a_variables_all_models[model_number][i-1][j] * self.ts_all_models[model_number][i-1][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT1')
                            # TYPE 2 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] <= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_all_models[model_number][i+1][j, 0]*(1-self.a_variables_all_models[model_number][i-1][j]), name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT2')
                            # TYPE 3 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= self.b_variables_all_models[model_number][i-1][j] * self.ts_all_models[model_number][i-1][j, 0], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT3')
                            # TYPE 4 Constraints for the whole network (except the output layer)
                            self.mip.addConstr(self.y_variables_all_models[model_number][i+1][j] >= gp.quicksum(weight[k] * self.y_variables_all_models[model_number][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts_all_models[model_number][i-1][j, 0] - self.upper_box_bounds_all_models[model_number][i+1][j, 0]) * self.b_variables_all_models[model_number][i-1][j], name=f'HLayer_{model_number}_{i+1}_{j}_Default_CT4')

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
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables_all_models[model_number][-2][k] for k in range(len(output_weight))) + output_bias + gp.quicksum(lin_skip_W[0, i]*self.y_variables_all_models[model_number][0][i] for i in range(lin_skip_W.shape[1])) == self.y_variables_all_models[model_number][-1][0], name=f'output_layer_{model_number}')
            # Linear Constraints for the output layer WIHTOUT lin_skip_layer: W*y + W_0*x
            else:
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables_all_models[model_number][-2][k] for k in range(len(output_weight))) + output_bias == self.y_variables_all_models[model_number][-1][0], name=f'output_layer_{model_number}')

        
         # --- Capacity Constraints --- # 
        for item_index in range(len(self.capacities)):
            self.mip.addConstr(gp.quicksum(self.y_variables_all_models[bidder_index][0][item_index] for bidder_index in range(len(self.models))) <= self.capacities[item_index], name=f'item_capacity_{item_index}')
        
        # --- Objective Declaration ---
        # self.mip.setObjective(self.y_variables[-1][0] - gp.quicksum(self.y_variables[0][i] * prices[i] for i in range(len(prices))), GRB.MAXIMIZE)
        self.mip.setObjective(gp.quicksum(self.scales[model_number] * self.y_variables_all_models[model_number][-1][0] for model_number in range(len(self.models))), GRB.MAXIMIZE)

        self.mip.update()
        if (verbose):
            self.mip.write('MVNN_generic_mip_SWMax_'+'_'.join(time.ctime().replace(':', '-').split(' '))+'.lp') # remark: if not updated mip.write() also calls mip.update()

        return



    def add_forbidden_allocation(self, forbidden_allocation, forbid_individual_bundles = True):
        """
        Takes as input a full allocation in the form [bundle_bidder_1, bundle_bidder_2, ...] and adds a constraint to the MIP that forbids this allocation.
        If forbid_individual_bundles is set to True: the allocation is forbidden for each bidder individually.
        """


        if self.forbid_single_bidder:
            # If we forbid a single bidder each time -> 
            # Store all allocations we need to forbid, and add the right part of the constraints when solving the MIP n-times (for each bidder)
            self.allocations_to_forbid.append(forbidden_allocation)
            return 


        # NEW: add constraint that the bundle is not already queried for generic bundles 
        # First define helper variables for absolute value

        for model_number, model in enumerate(self.models):
            bundle = forbidden_allocation[model_number]
            curr_len = len(self.diff_variables_all_models[model_number])   # only affects the naming of the variables
            self.diff_variables_all_models[model_number].append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle_{model_number}_{curr_len+1}_abs_helper_var_", vtype = GRB.INTEGER, lb = -GRB.INFINITY)) # models the difference y[model_number][0][i] - bundle[i]
            self.abs_variables_all_models[model_number].append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle_{model_number}_{curr_len+1}_abs_var_", vtype = GRB.INTEGER, lb = 0)) # models the absolute value of the difference y[model_number][0][i] - bundle[i]
        
            for m in range(len(bundle)):
                # Remark: access with [-1] always the last element of self.diff_variables and self.abs_variables
                self.mip.addConstr(self.diff_variables_all_models[model_number][-1][m] == (self.y_variables_all_models[model_number][0][m]-bundle[m]) , name=f'diff_CT_{model_number}_bundle_{curr_len+1}_item_{m}')
                self.mip.addConstr(self.abs_variables_all_models[model_number][-1][m] == gp.abs_(self.diff_variables_all_models[model_number][-1][m]), name=f'abs_CT_{model_number}_bundle_{curr_len+1}_item_{m}')

            if forbid_individual_bundles:
                self.mip.addConstr(gp.quicksum(self.abs_variables_all_models[model_number][-1][m] for m in range(len(bundle))) >= 1, name=f'alreadyQueried_bidder_{model_number}_bundle_{curr_len+1}')
        
        
        # self.mip.addConstr(gp.quicksum(self.abs_variables_all_models[model_number][-1][m] for m in range(len(bundle))) >= 1, name=f'alreadyQueried_bundle{curr_len+1}')
        if not forbid_individual_bundles:  # in that case, we forbid the allocation as a whole. only a single abs variable needs to be positive 
            # self.mip.addConstr(gp.quicksum(self.abs_variables[-1][m] for m in range(len(bundle))) >= 1, name=f'alreadyQueried_bundle{curr_len+1}')
            bundle = forbidden_allocation[0]  # we only need one bundle, since we only care about the length (number of items)
            self.mip.addConstr(gp.quicksum(self.abs_variables_all_models[model_number][-1][m] for model_number in range(len(self.models)) for m in range(len(bundle))) >= 1, name=f'alreadyQueried_allocation_{curr_len+1}')
        # -----------------------------

        self.mip.update()
        return
    
    def marginalize_bidder(self, bidder_index):

        self.marginalized_bidder = bidder_index  # store the marginalized bidder index

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
            # First define helper variables for absolute value
            curr_len = len(self.diff_variables_all_models[model_index])
            self.diff_variables_all_models[model_index].append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle_{model_index}_{curr_len+1}_abs_helper_var_", vtype = GRB.INTEGER, lb = -GRB.INFINITY)) # models the difference y[model_number][0][i] - bundle[i]
            self.abs_variables_all_models[model_index].append(self.mip.addVars([i for i in range(len(bundle))], name=f"forbidden_bundle_{model_index}_{curr_len+1}_abs_var_", vtype = GRB.INTEGER, lb = 0)) # models the absolute value of the difference y[model_number][0][i] - bundle[i]

            for m in range(len(bundle)):
                # Remark: access with [-1] always the last element of self.diff_variables and self.abs_variables
                self.mip.addConstr(self.diff_variables_all_models[model_index][-1][m] == (self.y_variables_all_models[model_index][0][m]-bundle[m]) , name=f'diff_CT_{model_index}_bundle_{curr_len+1}_item_{m}')
                self.mip.addConstr(self.abs_variables_all_models[model_index][-1][m] == gp.abs_(self.diff_variables_all_models[model_index][-1][m]), name=f'abs_CT_{model_index}_bundle_{curr_len+1}_item_{m}')

            self.mip.addConstr(gp.quicksum(self.abs_variables_all_models[model_index][-1][m] for m in range(len(bundle))) >= 1, name=f'alreadyQueried_bidder_{model_index}_bundle_{curr_len+1}')

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
            self.mip.Params.timeLimit = (timeLimit * 2) / len(self.models) # we solve n MIPs so need to divide the time limit by n
        self.mip.Params.MIPGap = MIPGap # Default 1e-04
        self.mip.Params.IntFeasTol = IntFeasTol # Default 1e-5
        self.mip.Params.FeasibilityTol = FeasibilityTol # Default 1e-6
        
        self.optimal_bundles = []
        self.mip.Params.OutputFlag = outputFlag

        if not self.forbid_single_bidder:
            self.start = timer()
            self.mip.optimize()
            self.end = timer()

            
            try:
                for model_index in range(len(self.models)):
                    bundle = []
                    for i in range(len(self.capacities)):
                        # if self.y_variables_all_models[model_index][0][i].x >= 0.99:
                        #     bundle.append(1)
                        # else:
                        #     bundle.append(0)
                        
                        bundle.append(np.int64(np.round(self.y_variables_all_models[model_index][0][i].x)))
                    self.optimal_bundles.append(bundle)
            except:
                # If an exception occurs during model optimization, generate IIS
                self.mip.computeIIS()
                # Write the IIS to a file
                self.mip.write('infeasible_model.ilp')
                    
                # self._print_info()
                raise ValueError('MIP did not solve successfully!')
            
        else: 
            try:
                self.start = timer() 
                # solve the unconstrained MIP once, get the unconstrained allocation out of it 
                self.mip.optimize()

                # get the unconstrained allocation
                unconstrained_allocation = []
                for model_index in range(len(self.models)):
                    bundle = []
                    for i in range(len(self.capacities)):
                        bundle.append(np.int64(np.round(self.y_variables_all_models[model_index][0][i].x)))
                    unconstrained_allocation.append(bundle)


                # solve the MIP n-times for each bidder with different allocation constraints
                # set_trace()
                for model_index in range(len(self.models)):
                    # set_trace()

                    bundles_to_forbid_single_bidder = [allocation[model_index] for allocation in self.allocations_to_forbid] # find all bundles to forbid for that bidder 
                    # only solve the MIP if the unconstrained allocation for that bidder is in the list of bundles to forbid
                    constraint_violation =  any(np.array_equal(unconstrained_allocation[model_index], arr) for arr in bundles_to_forbid_single_bidder)

                    if constraint_violation:
                        print(f'Found constraint violation, solving restricted MIP for bidder {model_index}')

                        self.forbid_bundles_single_bidder(bundles_to_forbid_single_bidder, model_index)
                        self.mip.optimize()
                    
                        bundle = []
                        for i in range(len(self.capacities)):
                            bundle.append(np.int64(np.round(self.y_variables_all_models[model_index][0][i].x)))
                        self.optimal_bundles.append(bundle)

                    else:
                        print(f'No constraint violation for bidder {model_index}, using unconstrained allocation')
                        self.optimal_bundles.append(unconstrained_allocation[model_index])

                self.end = timer()

            except:
                # If an exception occurs during model optimization, generate IIS
                self.mip.computeIIS()
                # Write the IIS to a file
                self.mip.write('infeasible_model.ilp')
                    
                # self._print_info()
                raise ValueError('MIP did not solve successfully!')


        self.optimal_bundles = np.array(self.optimal_bundles)
        # set_trace()
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