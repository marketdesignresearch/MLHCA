import numpy as np 



class Explicit100PercentUpperBoundIncludingDQS():

    def __init__(self, X_vqs, y_vqs, X_dqs, P_dqs):

        self.X_train = X_vqs   # _train refers to VQ type data
        self.y_train = y_vqs
        

        self.x_dqs = X_dqs   # _dqs refers to DQ type data
        self.p_dqs = P_dqs

        self.prepare_vq_data()
        self.precompute_ub_on_demands()
    

    def prepare_vq_data(self):
        """
        Sorts the value query date in ascending value order 
        """

        # add the zero bundle to the training data
        self.X_train = np.vstack([self.X_train, np.zeros(self.X_train.shape[1])])
        self.y_train = np.append(self.y_train, 0)

        
        # Sort the training data in ascending order value 
        argsort = np.argsort(self.y_train) 
        self.X_train = self.X_train[argsort]
        self.y_train = self.y_train[argsort]

    
    def calculate_value_query_UB(self, bundle):
        """
        Calculates the upper bound for a bundle based on the value query data 
        """

        for i in range(len(self.y_train)):
            if self.y_train[i] >= bundle:
                return self.y_train[i]
            
        return np.infty
    
    
    def precompute_ub_on_demands(self, iterations = None):
        """
        Precomputes a tighter 100% upper bound on the demand query points 
        """

        self.ubs_dqs = [self.calculate_value_query_UB(x_dq) for x_dq in self.x_dqs]  # start with the BOCA 100% UBs for all points

        if iterations is None:
            iterations = len(self.x_dqs) + 1  # TODO: check what the right number of iterations actually is. 

        for _ in range(iterations):  #  TODO: check what the right number of iterations actually is. 
            for i in range(len(self.x_dqs)):   
                alternative_values = [self.ubs_dqs[j] + np.dot(self.p_dqs[j], self.x_dqs[i] - self.x_dqs[j]) for j in range(len(self.x_dqs))]  # calculate the differences compared to all other values
                alternative_values.append(self.ubs_dqs[i])
                self.ubs_dqs[i] = min(alternative_values) # set its new value to the minimum of the alternatives
            

    def forward(self, x):
        """
        Performs the forward operation as discussed in the paper. 
        """

        
        vq_100_percent_ub = self.calculate_value_query_UB(x) # get the BOCA 100% prediction upper bound for x

        alternative_values = [self.ubs_dqs[j] + np.dot(self.p_dqs[j], x - self.x_dqs[j]) for j in range(len(self.x_dqs))]  # calculate the differences compared to all other values
        alternative_values.append(vq_100_percent_ub)
        return min(alternative_values) # set its new value to the minimum of the alternatives
