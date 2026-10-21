"""
File: hybridmodel.py
Description: Contains the hybrid model class, including
            methods for training and prediction. 
            

Author: Anndress07    
Last update: 1/12/2024

Usage:
            Accessed by main.py
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor


class HybridModel:
    """
    Contains the structure for the entire hybrid model, the decision tree and each linear regression for the leaf nodes,
    as well as the methods necessary for training ( fit() ) and prediction ( predict() )
    The fit() method is meant to be used first with the training data as parameters (both X_train and y_train). It will
    generate all objects needed for the training.
    After calling fit(), predict() can be used with the X_test (or the data to predict). It will yield
    linear_predictions, a list with the output for each sample in X_test.

    """

    def __init__(self):
        # print("Running the hybrid class")
        return

    def fit(self, X_train, y_train, LR_type, custom_params):
        """
        Trains the hybrid model, first creating the decision tree object using the tree() method. With the method
        classification() each sample is classified depending on the landing leaf node in the decision tree. Finally,
        the linear regression is performed for each leaf node. The relevant coefficients are saved in a dicitonary and
        used for later prediction methods.
        :param X_train: training dataset parameters
        :param y_train: training dataset outputs
        """
        if custom_params is not None:
            self.decision_tree_object = self.tree(X_train, y_train, custom_params = custom_params)
        else:
            self.decision_tree_object = self.tree(X_train, y_train, None)
        self.leaf_list, self.param_dict, self.output_dict = self.classification(self.decision_tree_object,
                                                                 X_train, y_train)
        self.LR_results = self.regressor(self.param_dict, self.output_dict, LR_type )

        print(f"\tFrom hybridmodel.fit() - Decision tree specs:")
        print(f"\t\tMax Tree Depth: {self.decision_tree_object.max_depth}")
        print(f"\t\tMax features: {self.decision_tree_object.max_features}")

        return

    def predict(self, X_test):
        """
        Performs the prediction for each sample in the dataset (X_test). First, an initial prediction with X_test and
        the decision tree is executed to see at which leaf node each sample lands. The apply() lists contain the node
        for each sample.
        For each sample in the apply() list, the proper model is searched for in the coefficients results dictionary.
        When the correct model is found, we can gather its coefficients and intercept for the prediction as y = X*β + ε
        where
                X: the feature vector for given sample
                β: the coefficients vector for the sample's leaf node
                ε: the intercept for the sample's leaf node
                y: resulting prediction for given sample
        Each y result is attached to the linear_predictions DataFrame
        :param X_test: training/to predict dataset parameters
        :return linear_predictions: DataFrame that contains each prediction value of X_test
        """
        # self.linear_predictions = pd.DataFrame(columns=['node_id','idx on X_test','y_pred','opl_pred'])
        self.linear_predictions = []
        if hasattr(self, 'decision_tree_object'):
            self.leaf_test_list  = self.decision_tree_object.apply(X_test)
            print(f"lenght of test list {len(X_test)}")
            print(f"lenght of apply list {len(self.leaf_test_list)}")
            for leaf_node in range(len(X_test)):
                leaf_id_value = self.leaf_test_list[leaf_node]
                for model in self.LR_results:
                    if model['Model: '] == leaf_id_value:
                        current_object = model["Object: "]
                        current_X = X_test.iloc[leaf_node].to_frame().T.values # w/o feature names
                        y_lr_pred = current_object.predict(current_X)

                        #### debug
                        # col_names = [' Fanout', ' Cap', ' Slew', ' Delay', 'X_drive', 'Y_drive', 'X_sink',
                        #              'Y_sink', 'C_drive', 'C_sink', 'X_context', 'Y_context', 'σ(X)_context',
                        #              'σ(Y)_context', 'Drive_cell_size', 'Sink_cell_size', 'Label Delay']
                        # sample_to_test = [23960, 25870, 56097, 42310, 15001]
                        # if leaf_node in sample_to_test:
                            # print(f"Description of sample {leaf_node} of node {leaf_id_value}. ")
                            # print(f"\t y_pred = {y_lr_pred[0]}")
                            # for i in range(len(current_object.coef_)):
                                # print(
                                #     f"\tParameter {i:<2}: {col_names[i]:<15}, coef: {current_object.coef_[i]:<12.4f}, sample: {current_X[0][i]:<11.2f}"
                                #     f"result: {current_object.coef_[i] * current_X[0][i]:<12.2f}")
                        #print(f"y_lr_pred: {y_lr_pred}")
                        #y_lr_series = pd.Series([y_lr_pred], index=[leaf_node], name='linear predictions')
                        # TODO: return a numpy array with y_pred instead
                        self.linear_predictions.append(y_lr_pred[0])
                        # self.linear_predictions = np.array(self.linear_predictions)
                        # self.linear_predictions = self.linear_predictions._append({'node_id': model['Model: '],
                        #                                                           'idx on X_test': leaf_node,
                        #                                                            "y_pred": y_lr_pred[0]
                        #                                                            ,"opl_pred": current_X[0][3]
                        #                                                            },
                        #                                                           ignore_index= True)
                        #print("===================================\n\n")
        else:
            print("Error: fit() must be called before predict()")
        return self.linear_predictions



    def tree(self, X_train, y_train, custom_params = None):
        """
        Builds the initial decision tree used for sample classification
        :param training_data: dataset to train the tree
        :return dtr: the decision tree object
        """
        if custom_params is None:
            self.dtr = DecisionTreeRegressor(max_depth=9, max_features=15, random_state=10)
        else:
            self.dtr = DecisionTreeRegressor(max_depth=custom_params[0], max_features=custom_params[1])

        self.dtr.fit(X_train, y_train)
        return self.dtr

    def classification(self, dtr, X_train, y_train):
        """
        Classifies each sample depending on the leaf node of the decision tree they land in
        using the apply() method. Creates a dictionary where each node contains all its samples in a list
            leaf_params_dict = {ID1: [ [sample1],[sample2]...], ID2: [ [sample3],[sample4]...]... }
            leaf_result_dict = {ID1: [ [y1, y2]...]. ID2: [y3, y4]...}
        :param dtr: decision tree structure
        :param X_train: parameter training set
        :param y_train: prediction set
        :return leaf_params_dict: Contains all leaf nodes with its grouped sample parameters:
        :return leaf_result_dict: Contains all leaf nodes with its grouped sample predictions
        """
        self.leaf_sample_list = dtr.apply(X_train)

        self.leaf_params_dict = {}
        self.leaf_result_dict = {}

        for leaf_node in range(len(X_train)):
            leaf_id_value = self.leaf_sample_list[leaf_node]
            if leaf_id_value not in self.leaf_params_dict:
                self.leaf_params_dict[leaf_id_value] = []
            if leaf_id_value not in self.leaf_result_dict:
                self.leaf_result_dict[leaf_id_value] = []

            self.leaf_params_dict[leaf_id_value].append(X_train.iloc[leaf_node].tolist())
            self.leaf_result_dict[leaf_id_value].append(y_train.iloc[leaf_node])

        ### DEBUG
        # nodo_prueba_x = []
        # for leaf_node in range(len(X_test)):
        #     if leaf_sample_list[leaf_node] == 598:
        #         nodo_prueba_x.append(X_test.iloc[leaf_node].tolist())
        #         nodo_prueba_x.append(y_train.iloc[leaf_node].tolist())
        # dfx = pd.DataFrame(nodo_prueba_x)
        # dfx.to_csv("max_error.csv", index=False)
        return self.leaf_sample_list, self.leaf_params_dict, self.leaf_result_dict

    def regressor(self, leaf_params_dict, leaf_result_dict, linear_type = 0):
        """
        Implements the linear regression for each leaf node generated in the decision tree,
        that is, every entry on the dictionary leaf_params_dict and leaf_result_dict

        :param leaf_params_dict: Dictionary with all leaf nodes and its classified samples (parameters)
        :param leaf_result_dict: Dictionary with all leaf nodes and its classified samples (outputs)
        :return LR_results: List of dictionaries with relevant information of the linear regression
            LR_results = [{"Model": ID of the leaf node,
                          "Coefficients: ": LR.coef_,
                          "Intercept: ": LR.intercept_,
                          "RMSE ML: ": difference between Label Delay and the prediction
                          "RMSE OpenLane: ": difference between Delay and Label Delay
                              }, ...]
        """
        self.LR_results = []
        counter_progress = 1
        # print(f"Linear type is {linear_type}")
        for key, val in leaf_params_dict.items():
            # print(f"Executing n#{counter_progress} out of {len(leaf_params_dict)}")
            # print(f"Node ID: {key} \t\t Value: ", end='')
            # print(f"Depth of val: {len(val)}")
            counter_progress = counter_progress + 1

            '''
                Contador para detener la ejecución 
            '''
            # counter_progress = counter_progress + 1
            # if counter_progress > 3:
            #     break

            if (len(val) > 0):
                X_LR = leaf_params_dict[key]
                y_LR = leaf_result_dict[key]

                if linear_type == 0:
                    LR = linear_model.LinearRegression()
                elif linear_type == 1:
                    LR = Ridge(alpha=1.0)

                LR.fit(X_LR, y_LR)
                resultado_LR = {"Model: ": key,"Object: ": LR}
                self.LR_results.append(resultado_LR)


        return self.LR_results
