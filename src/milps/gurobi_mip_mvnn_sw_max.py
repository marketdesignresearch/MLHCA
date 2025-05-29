import time
from timeit import default_timer as timer
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from pdb import set_trace

class  GUROBI_MIP_MVNN_MULTIPLE_BIDDERS_SW_MAX:

    def __init__(self, models, scales, item_capacities, SATS_domain, GSVM_national_bidder_goods_of_interest):
        # MVNN PARAMETERS
        self.models = models  # List of MVNN TORCH MODELS
        self.scales = scales  # List of scales for each model
        self.item_capacities = item_capacities  # List of item capacities
        self.ts = [[layer.ts.data.cpu().detach().numpy().reshape(-1, 1) for layer in model.layers] for model in models]
        # MIP VARIABLES
        self.y_variables = []  # CONT VARS 1
        self.a_variables = []  # BINARY VARS 1
        self.b_variables = []  # BINARY VARS 2
        self.case_counter = {'Case1': 0, 'Case2': 0, 'Case3': 0, 'Case4': 0, 'Case5': 0}
        # SATS PARAMETERS
        self.SATS_domain = SATS_domain
        self.GSVM_national_bidder_goods_of_interest = GSVM_national_bidder_goods_of_interest

    
    def calc_preactivated_box_bounds(self, model, model_index,input_upper_bound=1, input_lower_bound=0, verbose=False):
        # BOX-bounds for y variable (preactivated!!!!) as column vectors

        # Initialize
        upper_box_bounds = [np.array([input_upper_bound] * model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)]
        lower_box_bounds = [np.array([input_lower_bound] * model.layers[0].in_features, dtype=np.int64).reshape(-1, 1)]

        # Propagate through Network
        for i, layer in enumerate(model.layers):
            W = layer.weight.data.cpu().detach().numpy()
            b = layer.bias.data.cpu().detach().numpy().reshape(-1, 1)

            # -------------------
            if i == 0:
                upper_box_bounds.append(W @ upper_box_bounds[-1] + b)
                lower_box_bounds.append(W @ lower_box_bounds[-1] + b)
            else:
                t = self.ts[model_index][i-1]
                upper_box_bounds.append(W @ self.phi(upper_box_bounds[-1], t) + b)
                lower_box_bounds.append(W @ self.phi(lower_box_bounds[-1], t) + b)
            # -------------------

        if verbose:
            print('Upper Box Bounds:')
            print(upper_box_bounds)
            print('Lower Box Bounds:')
            print(lower_box_bounds)
        
        return upper_box_bounds, lower_box_bounds

    def phi(self, x, t):
        # Bounded ReLU (bReLU) activation function for MVNNS with cutoff t
        return np.minimum(t, np.maximum(0, x)).reshape(-1, 1)

    def generate_mip(self, MIPGap=None, verbose=False):
        self.mip = gp.Model("MVNN_MIP_MULTIPLE_BIDDERS")

        # Add IntFeasTol, primal feasibility
        if MIPGap:
            self.mip.Params.MIPGap = MIPGap

        # Calculate preactivated box bounds for each model
        self.upper_box_bounds_list = []
        self.lower_box_bounds_list = []
        for model_index, model in enumerate(self.models):
            upper_box_bounds, lower_box_bounds = self.calc_preactivated_box_bounds(model, model_index= model_index,verbose=verbose)
            self.upper_box_bounds_list.append(upper_box_bounds)
            self.lower_box_bounds_list.append(lower_box_bounds)

        # --- Variable declaration --- # 
        for model_index, model in enumerate(self.models):
            self.y_variables.append([self.mip.addVars([i for i in range(len(self.item_capacities))], name=f"x_{model_index}_", vtype=GRB.BINARY)])  # input variables

            for (i, layer) in enumerate(model.layers):
                tmp_y_variables = []
                for j in range(len(layer.weight.data)):
                    tmp_y_variables.append(self.mip.addVar(name=f'y_{model_index}_{i+1}_{j}', vtype=GRB.CONTINUOUS, lb=0, ub=self.ts[model_index][i][j, 0]))
                self.y_variables[model_index].append(tmp_y_variables)
                self.a_variables.append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'a_{model_index}_{i+1}_', vtype=GRB.BINARY))
                self.b_variables.append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'b_{model_index}_{i+1}_', vtype=GRB.BINARY))

            layer = model.output_layer
            self.y_variables[model_index].append(self.mip.addVars([j for j in range(len(layer.weight.data))], name=f'y_output_{model_index}_', vtype=GRB.CONTINUOUS, lb=0))

        # MVNN Constraints
        for model_index, model in enumerate(self.models):
            for (i, layer) in enumerate(model.layers):
                for (j, weight) in enumerate(layer.weight.data):
                    if self.lower_box_bounds_list[model_index][i+1][j, 0] >= self.ts[model_index][i][j, 0]:
                        self.y_variables[model_index][i+1][j] = self.ts[model_index][i][j, 0]
                        self.case_counter['Case1'] += 1
                    elif self.upper_box_bounds_list[model_index][i+1][j, 0] <= 0:
                        self.y_variables[model_index][i+1][j] = 0
                        self.case_counter['Case2'] += 1
                    elif (self.lower_box_bounds_list[model_index][i+1][j, 0] >= 0 and self.lower_box_bounds_list[model_index][i+1][j, 0] <= self.ts[model_index][i][j, 0]) and (self.upper_box_bounds_list[model_index][i+1][j, 0] >= 0 and self.upper_box_bounds_list[model_index][i+1][j, 0] <= self.ts[model_index][i][j, 0]):
                        self.y_variables[model_index][i+1][j] = gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j]
                        self.case_counter['Case3'] += 1
                    elif self.lower_box_bounds_list[model_index][i+1][j, 0] >= 0:
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= self.ts[model_index][i][j, 0], name=f'HLayer_{model_index}_{i+1}_{j}_Case4_CT1')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_index}_{i+1}_{j}_Case4_CT2')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= self.b_variables[i][j] * self.ts[model_index][i][j, 0], name=f'HLayer_{model_index}_{i+1}_{j}_Case4_CT3')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts[model_index][i][j, 0] - self.upper_box_bounds_list[model_index][i+1][j, 0]) * self.b_variables[i][j], name=f'HLayer_{model_index}_{i+1}_{j}_Case4_CT4')
                        self.case_counter['Case4'] += 1
                    elif self.upper_box_bounds_list[model_index][i+1][j, 0] <= self.ts[model_index][i][j, 0]:
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= self.a_variables[i][j] * self.ts[model_index][i][j, 0], name=f'HLayer_{model_index}_{i+1}_{j}_Case5_CT1')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_list[model_index][i+1][j, 0]*(1-self.a_variables[i][j]), name=f'HLayer_{model_index}_{i+1}_{j}_Case5_CT2')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= 0, name=f'HLayer_{model_index}_{i+1}_{j}_Case5_CT3')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j], name=f'HLayer_{model_index}_{i+1}_{j}_Case5_CT4')
                        self.case_counter['Case5'] += 1
                    else:
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= self.a_variables[i][j] * self.ts[model_index][i][j, 0], name=f'HLayer_{model_index}_{i+1}_{j}_Default_CT1')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] <= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j] - self.lower_box_bounds_list[model_index][i+1][j, 0]*(1-self.a_variables[i][j]), name=f'HLayer_{model_index}_{i+1}_{j}_Default_CT2')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= self.b_variables[i][j] * self.ts[model_index][i][j, 0], name=f'HLayer_{model_index}_{i+1}_{j}_Default_CT3')
                        self.mip.addConstr(self.y_variables[model_index][i+1][j] >= gp.quicksum(weight[k] * self.y_variables[model_index][i][k] for k in range(len(weight))) + layer.bias.data[j] + (self.ts[model_index][i][j, 0] - self.upper_box_bounds_list[model_index][i+1][j, 0]) * self.b_variables[i][j], name=f'HLayer_{model_index}_{i+1}_{j}_Default_CT4')

            output_weight = model.output_layer.weight.data[0]
            if (model.output_layer.bias is not None):
                output_bias = model.output_layer.bias.data
            else:
                output_bias = 0

            if output_bias != 0:
                raise ValueError('output_bias is not 0')

            if hasattr(model, 'lin_skip_layer'):
                lin_skip_W = model.lin_skip_layer.weight.detach().cpu().numpy()
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[model_index][-2][k] for k in range(len(output_weight))) + output_bias + gp.quicksum(lin_skip_W[0, i]*self.y_variables[model_index][0][i] for i in range(lin_skip_W.shape[1])) == self.y_variables[model_index][-1][0], name=f'output_layer_{model_index}')
            else:
                self.mip.addConstr(gp.quicksum(output_weight[k] * self.y_variables[model_index][-2][k] for k in range(len(output_weight))) + output_bias == self.y_variables[model_index][-1][0], name=f'output_layer_{model_index}')

        # --- Capacity Constraints --- # 
        for item_index in range(len(self.item_capacities)):
            self.mip.addConstr(gp.quicksum(self.y_variables[bidder_index][0][item_index] for bidder_index in range(len(self.models))) <= self.item_capacities[item_index], name=f'item_capacity_{item_index}')

        # --- Domain-specific constraints --- #
        if self.SATS_domain == 'GSVM':
            for model_index in range(len(self.models)):
                if model_index == 6:
                    GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement = [i not in self.GSVM_national_bidder_goods_of_interest for i in range(len(self.item_capacities))]
                    self.mip.addConstr(gp.quicksum(self.y_variables[model_index][0][i] * GSVM_national_bidder_goods_of_interest_one_hot_encoding_complement[i] for i in range(len(self.item_capacities))) == 0, name=f"GSVM_CT_NationalBidder_{model_index}")
                else:
                    self.mip.addConstr(gp.quicksum(self.y_variables[model_index][0][i] for i in range(len(self.item_capacities))) <= 4, name=f"GSVM_CT_RegionalBidder_{model_index}")

        # Objective Declaration
        self.mip.setObjective(gp.quicksum(self.scales[model_index] * self.y_variables[model_index][-1][0] for model_index in range(len(self.models))), GRB.MAXIMIZE)

        if verbose:
            self.mip.write('MVNN_mip_multiple_bidders_'+'_'.join(time.ctime().replace(':', '-').split(' '))+'.lp')


    def add_forbidden_allocation(self, forbidden_allocation):
        """
        Adds a constraint to the MIP model to exclude a specific allocation.

        :param forbidden_allocation: A list of lists where each sublist corresponds to the allocation of a bidder.
                                    Each sublist contains 1s and 0s indicating the presence or absence of items.
        """
        print('add forbidden allocation called')
        set_trace()
        if len(forbidden_allocation) != len(self.models):
            raise ValueError("The forbidden allocation must have the same number of sublists as there are models.")
        
        if any(len(allocation) != len(self.item_capacities) for allocation in forbidden_allocation):
            raise ValueError("Each sublist in the forbidden allocation must have the same length as the item capacities list.")
        
        expr_list = []
        for model_index, allocation in enumerate(forbidden_allocation):
            for item_index, item in enumerate(allocation):
                if item == 1:
                    expr_list.append(self.y_variables[model_index][0][item_index])
                else:
                    expr_list.append(1 - self.y_variables[model_index][0][item_index])
        
        self.mip.addConstr(gp.quicksum(expr_list) <= len(expr_list) - 1, name='forbidden_allocation')
        self.mip.update()


    def solve_mip(self, outputFlag=False, verbose=True, timeLimit=np.inf, MIPGap=1e-04, IntFeasTol=1e-5, FeasibilityTol=1e-6):
        if not verbose:
            self.mip.Params.LogToConsole = 0
            self.mip.Params.OutputFlag = 0

        self.mip.Params.timeLimit = timeLimit
        self.mip.Params.MIPGap = MIPGap
        self.mip.Params.IntFeasTol = IntFeasTol
        self.mip.Params.FeasibilityTol = FeasibilityTol

        self.start = timer()
        self.mip.Params.OutputFlag = outputFlag
        self.mip.optimize()
        self.end = timer()

        self.optimal_schedules = []
        try:
            for model_index in range(len(self.models)):
                schedule = []
                for i in range(len(self.item_capacities)):
                    if self.y_variables[model_index][0][i].x >= 0.99:
                        schedule.append(1)
                    else:
                        schedule.append(0)
                self.optimal_schedules.append(schedule)
        except:
            self._print_info()
            raise ValueError('MIP did not solve successfully!')

        if verbose:
            self._print_info()

        return np.array(self.optimal_schedules)

    def _print_info(self):
        print('*' * 30)
        print('MIP INFO:')
        print('-' * 30)
        print(f'Name: {self.mip.ModelName}')
        print(f'Goal: {self._model_sense_converter(self.mip.ModelSense)}')
        print(f'Objective: {self.mip.getObjective()}')
        print(f'Number of variables: {self.mip.NumVars}')
        print(f' - Binary {self.mip.NumBinVars}')
        print(f'Number of linear constraints: {self.mip.NumConstrs}')
        print(f'Primal feasibility tolerance for constraints: {self.mip.Params.FeasibilityTol}')
        print(f'Integer feasibility tolerance: {self.mip.Params.IntFeasTol}')
        print(f'Relative optimality gap: {self.mip.Params.MIPGap}')
        print(f'Time Limit: {self.mip.Params.TimeLimit}')
        print('')
        print('MIP SOLUTION:')
        print('-' * 30)
        print(f'Status: {self._status_converter(self.mip.status)}')
        print(f'Elapsed time in sec: {self.end - self.start}')
        print(f'Reached relative optimality gap: {self.mip.MIPGap}')
        for model_index in range(len(self.models)):
            print(f'Optimal Allocation for model {model_index}: {self.optimal_schedules[model_index]}')
        print(f'Objective Value: {self.mip.ObjVal}')
        print(f'Number of stored solutions: {self.mip.SolCount}')
        print('IA Case Statistics:')
        for k, v in self.case_counter.items():
            print(f' - {k}: {v}')
        print('*' * 30)

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

