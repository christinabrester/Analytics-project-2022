# MLP is trained to nowcast one of the PQ variables
# A number of achitectures are trained and evaluated on the validation data to be compared with each other
# This script might be run using 'parallel_run.py' -> to train models for different PQ variables and phases in parallel

# Line 242: define an experiment (ventilation system or main distr board)
# Lines 268-273: remove/add inputs, if needed


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys

from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K

def metric_IA(y_true,y_pred):
    sse = K.sum(K.square(y_true - y_pred))
    ia = K.sum(K.square(K.abs(y_true - K.mean(y_true)) + K.abs(y_pred - K.mean(y_true))))

    ia = 1. - sse/ia
    return ia

def IA(true, predicted):  
    true = true.ravel()
    predicted = predicted.ravel()
    return 1 - np.sum((true-predicted)**2)/np.sum((np.abs(predicted-np.mean(true))+np.abs(true-np.mean(true)))**2)

def RMSE(true, predicted):
    true = true.ravel()
    predicted = predicted.ravel()
    return np.sqrt(np.mean((true-predicted)**2))

def MAE(true, predicted):
    true = true.ravel()
    predicted = predicted.ravel()
    return np.mean(np.abs(true-predicted))

def training_validation_split(timestamps, percent_valid):
    days_all = pd.to_datetime(timestamps).apply(lambda x: x.date())
    validation_days = np.random.choice(list(set(days_all.values)), int(percent_valid*len(set(days_all.values))), replace = False)
        
    validation_index = days_all.apply(lambda x: x in validation_days)
    training_index = days_all.apply(lambda x: x not in validation_days)
    
    return training_index, validation_index


class NeuralNet(ABC):
    MAX_EPOCHS = 1000
    BATCH_SIZE = 128
    LOSS = 'mean_absolute_error'
    METRICS = [metric_IA, 'mean_squared_error']
    LEARNING_RATE = 1e-3
    ACTIVATION = 'tanh'
    NNEURONS = [8]

    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', NeuralNet.BATCH_SIZE)
        self.loss = kwargs.get('loss', NeuralNet.LOSS)
        self.metrics = kwargs.get('metrics', NeuralNet.METRICS)
        self.max_epochs = kwargs.get('metrics', NeuralNet.MAX_EPOCHS)
        self.learning_rate = kwargs.get('learning_rate', NeuralNet.LEARNING_RATE)
        self.activation = kwargs.get('activation', NeuralNet.ACTIVATION)
        self.nneurons = kwargs.get('nneurons', NeuralNet.NNEURONS[:])

        self.model = Sequential()
        #self.random_weights = None # used to re-initialize the model in new runs


    @abstractmethod
    def build_model(self):
        pass

    def train_model_with_valid(self, data_train_x, data_train_y, data_val_x, data_val_y):
        '''
        A training mode to define an optimal number of epochs using validation data.
        Returns a tuple (opt_epochs, opt_val_loss)
        '''

        #self.model.set_weights(self.random_weights) # re-initialize the model
        
        opt = Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            loss = self.loss,
            optimizer = opt,
            metrics =  self.metrics
        )
        
        earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=0, mode='min', restore_best_weights=True)
        
        self.model.fit(
            data_train_x, data_train_y,
            verbose=0,
            shuffle=True,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            validation_data=(data_val_x, data_val_y),
            callbacks=[earlystopping]
        )
        
        hist = self.model.history.history['val_loss']
        n_epochs_best = np.argmin(hist)
        scores_validation = self.model.evaluate(data_val_x, data_val_y, verbose=0, batch_size=data_val_x.shape[0])
        
        return n_epochs_best
    
    def train_model(self, data_x, data_y, opt_epochs):
        '''
        A training mode without validation when the optimal number of epochs is known.
        '''
        #self.model.set_weights(self.random_weights) # re-initialize the model
        
        opt = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss = self.loss,
            optimizer = opt,
            metrics =  self.metrics
        )
        self.model.fit(
            data_x, data_y,
            verbose=0,
            shuffle=True,
            batch_size=self.batch_size,
            epochs=int(opt_epochs),
            validation_split = 0.0
        )

    def predict(self, data_x):
        return self.model.predict(data_x, batch_size = data_x.shape[0])
            

class MLP(NeuralNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_model(self, input_cols):
        self.model = Sequential()
        for i, nn in enumerate(self.nneurons):
            if i == 0:
                self.model.add(Dense(nn, input_shape=(input_cols,), activation = self.activation))
            else:
                self.model.add(Dense(nn, activation = self.activation))
        self.model.add(Dense(1, activation = 'linear'))

        #self.random_weights = self.model.get_weights() # save random weights to re-initialize the model in new runs


def modeling(training_data, training_timestamps, test_data, inputs, output, NN):
    
    training_data_inputs = training_data.loc[:, inputs].copy()
    test_data_inputs = test_data.loc[:, inputs].copy()

    training_data_output = pd.DataFrame(training_data.loc[:, output].copy(), columns = [output])
    test_data_output = pd.DataFrame(test_data.loc[:, output].copy(), columns = [output])
    
    # 1. Train scalers
    scaler_inputs = MinMaxScaler()
    scaler_inputs.fit(training_data_inputs)
    
    training_data_inputs_scaled = scaler_inputs.transform(training_data_inputs)
    test_data_inputs_scaled = scaler_inputs.transform(test_data_inputs)
    #---------------------------------------------------------------------
    scaler_output = MinMaxScaler()
    scaler_output.fit(training_data_output)
    
    training_data_output_scaled = scaler_output.transform(training_data_output)
    test_data_output_scaled = scaler_output.transform(test_data_output)
    #---------------------------------------------------------------------
    
    # 2. Validation performance
    ia_valid = 0
    rmse_valid = 0
    mae_valid = 0
    
    n_folds = 5
    length = len(training_data)
    arr = np.arange(n_folds)
    array_folds = np.append(np.repeat(arr, int(length/n_folds)), np.repeat(np.array([n_folds-1]), length - n_folds*int(length/n_folds)))    
        
    opt_epochs_list = []
    for fold in range(n_folds):
        
        X_train = training_data_inputs_scaled[array_folds != fold]
        X_valid = training_data_inputs_scaled[array_folds == fold]
        y_train = training_data_output_scaled[array_folds != fold]
        y_valid = training_data_output_scaled[array_folds == fold]
        
        NN.build_model(X_train.shape[1])
        
        opt_epochs = NN.train_model_with_valid(X_train, y_train, X_valid, y_valid)
        
        opt_epochs_list.append(opt_epochs)      

        pred_valid = NN.predict(X_valid).reshape(-1, 1)
        pred_valid = scaler_output.inverse_transform(pred_valid)
        y_valid = scaler_output.inverse_transform(y_valid)
    
        ia_valid = ia_valid + IA(y_valid, pred_valid)/n_folds
        rmse_valid = rmse_valid + RMSE(y_valid, pred_valid)/n_folds
        mae_valid = mae_valid + MAE(y_valid, pred_valid)/n_folds

        
    opt_epochs = np.mean(opt_epochs_list)
    #---------------------------------------------------------------------

    # 3. Training and test performance
    
    NN.build_model(training_data_inputs_scaled.shape[1])
    NN.train_model(training_data_inputs_scaled, training_data_output_scaled.ravel(), opt_epochs)
    
    pred_train = NN.predict(training_data_inputs_scaled).reshape(-1, 1)
    
    pred_train = scaler_output.inverse_transform(pred_train)
    training_data_output_scaled = scaler_output.inverse_transform(training_data_output_scaled)
    
    ia_train = IA(training_data_output_scaled, pred_train)
    rmse_train = RMSE(training_data_output_scaled, pred_train)
    mae_train = MAE(training_data_output_scaled, pred_train)
    
    pred_test = NN.predict(test_data_inputs_scaled).reshape(-1, 1)
    pred_test = scaler_output.inverse_transform(pred_test)

    test_data_output = scaler_output.inverse_transform(test_data_output_scaled)
    
    ia_test = IA(test_data_output, pred_test)
    rmse_test = RMSE(test_data_output, pred_test)
    mae_test = MAE(test_data_output, pred_test)
    
    
    result = {"ia_train": ia_train, "rmse_train": rmse_train, "mae_train": mae_train, 
              "ia_valid": ia_valid, "rmse_valid": rmse_valid, "mae_valid": mae_valid, 
              "ia_test": ia_test, "rmse_test": rmse_test, "mae_test": mae_test}

    return pred_train, pred_test, result, opt_epochs


if __name__ == "__main__":

    exp_name = 'main_distribution_board_2018_2020' # or exp_name = 'ventilation_system_2018_2020'
    
    training_data = pd.read_csv('{}/measurement_data_2018_2019.csv'.format(exp_name))
    test_data = pd.read_csv('{}/measurement_data_2019_2020.csv'.format(exp_name))

    training_data_noNaN = training_data.dropna().copy()
    test_data_noNaN = test_data.dropna().copy()

    column_names = training_data.columns

    column_names_L1 = [var for var in column_names if 'L1' in var ]
    column_names_L2 = [var for var in column_names if 'L2' in var ]
    column_names_L3 = [var for var in column_names if 'L3' in var ]

    column_names_dict = {'L1': column_names_L1, 'L2': column_names_L2, 'L3': column_names_L3}


    # Experiment
    phase = sys.argv[1]
    output_original = sys.argv[2] # TDU ITD Q1act

    output = output_original + phase
    inputs = column_names_dict[phase].copy()
    inputs.remove(output)
    inputs.remove('Q1{}'.format(phase))
    inputs.remove('I1{}'.format(phase))
    inputs.append('U0U1')
    inputs.append('U2U1')
    inputs.append('Freq')

    training_timestamps = training_data_noNaN.loc[:, 'Timestamp']
    
    # Test different MLP architectures
    hidden_layers = [ [4], [4,4], [4,4,4],
                                    [8], [8,8], [8,8,8],
                                    [16], [16,16], [16,16,16],
                                    [32], [32,32], [32,32,32],
                                    [64,64,64], [128,128,128], 
                                    [64, 32, 16], [128,64,32] ]

    for exp_run in range(10):
        results = pd.DataFrame(columns = ["ia_train", "rmse_train", "mae_train", 
                                          "ia_valid", "rmse_valid", "mae_valid", 
                                          "ia_test", "rmse_test", "mae_test", "nneurons", "epochs"])
        
        print(output_original, ' ', phase)

        for nneurons in hidden_layers:
            print(nneurons)

            model = MLP(nneurons = nneurons) 
            pred_train, pred_test, result, epochs = modeling(training_data_noNaN, training_timestamps, test_data_noNaN, inputs, output, model)

            result['nneurons'] = nneurons
            result['epochs'] = epochs

            results = results.append(result, ignore_index=True)

        results.to_csv("{}/results_mlp_{}_{}_run_{}.csv".format(exp_name, output_original, phase, exp_run))