import numpy as np
import pickle
from sklearn import linear_model
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import inspect
import helpers
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPool1D, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
import copy
import utilsforminds

# def create_CNN_model(param_dict):
#     model = Sequential()
#     model.add(Conv1D(filters = param_dict['conv_1']['filters'], kernel_size = param_dict['conv_1']['kernel_size'], activation = 'relu', input_shape = (self.x_length[x_form], 1), padding = 'valid'))
#     model.add(MaxPool1D(pool_size=2))
#     model.add(Conv1D(filters = param_dict['conv_2']['filters'], kernel_size = param_dict['conv_2']['kernel_size'], activation = 'relu', padding = 'valid'))
#     model.add(MaxPool1D(pool_size=2))
#     model.add(Flatten())
#     model.add(Dense(param_dict['dense_1']['neurons']))
#     model.add(Dropout(rate = param_dict['dense_1']['drop_rate']))
#     model.add(Dense(param_dict['dense_2']['neurons']))
#     model.add(Dropout(rate = param_dict['dense_2']['drop_rate']))
#     model.add(Dense(param_dict['dense_3']['neurons']))
#     model.add(Dropout(rate = param_dict['dense_3']['drop_rate']))
#     model.add(Dense(1))
#     model.compile(optimizer = 'adam', loss = 'mse')
#     return model
rmse_sklearn = make_scorer(utilsforminds.math.get_RMSE, greater_is_better= False)

class Other_models_tester():
    def __init__(self, X_all, target_arr_training, target_arr_test, W, info_dct, target_original_arr, debug = False, min_max_enriched = True, raw_enriched = ['raw', 'enriched'], x_timepoint_idx= 0, y_timepoint_idx= 0, features_range_dict = None, add_dict = None):
        """

        Parameters
        ----------
        raw_enriched = ['raw', 'enriched']: list
            Kinds of data type to train and test.
        """

        if add_dict is None: add_dict = {}
        if isinstance(X_all, str):
            self.X_all = np.load(X_all)
            self.W = np.load(W)
            self.target_arr_training = np.load(target_arr_training)
            self.target_arr_test = np.load(target_arr_test)
            target_original_arr = np.load(target_original_arr) ## only used in debugging.
            with open(info_dct, 'rb') as pickle_file:
                self.data_info_dct = pickle.load(pickle_file)
        else:
            self.X_all = np.copy(X_all)
            self.W = np.copy(W)
            self.target_arr_training = np.copy(target_arr_training)
            self.target_arr_test = np.copy(target_arr_test)
            self.data_info_dct = copy.deepcopy(info_dct)
        if self.data_info_dct["dataset_kind"] == "traffic":
            if "img_paths" in add_dict.keys():
                self.img_paths = add_dict["img_paths"]
            else:
                with open(add_dict["img_paths"], 'rb') as pickle_file:
                    self.img_paths = pickle.load(pickle_file)
            
        assert(len(self.X_all.shape) == 3 and self.X_all.shape[0] == self.W.shape[0])
        assert(self.W.shape[1] == self.data_info_dct['d'] and self.W.shape[2] == self.data_info_dct['r_0'])
        self.modality = self.target_arr_training.shape[1]
        assert(self.modality == self.target_arr_test.shape[1] and self.modality == target_original_arr.shape[1])
        self.num_of_patients = self.X_all.shape[0]
        self.num_of_training_set = self.data_info_dct['num_of_training_set']
        self.target_labels = self.data_info_dct['target_label']
        assert(self.modality == len(self.target_labels))
        self.r_0 = self.data_info_dct['r_0']
        self.x_length = {'raw': self.data_info_dct['d'], 'enriched': self.r_0}
        self.output_length = {'training': self.num_of_training_set, 'test': self.num_of_patients - self.num_of_training_set}
        self.raw_enriched = raw_enriched
        self.readable_raw_enriched = {'raw': 'Original Representation', 'enriched': 'Enriched Representation'}
        self.features_range_dict = features_range_dict
        if debug:
            for i in range(self.num_of_training_set):
                assert(np.array_equal(target_original_arr[self.data_info_dct['idc_patients_shuffled'][i]], self.target_arr_training[i]))
            for i in range(self.num_of_patients - self.num_of_training_set):
                assert(np.array_equal(target_original_arr[self.data_info_dct['idc_patients_shuffled'][self.num_of_training_set + i]], self.target_arr_test[i]))
        
        #%% Set y
        self.y_dct = {}
        for label in self.target_labels:
            self.y_dct[label] = {}
        # self.y_dct['BL_RAVLT_TOTAL']['train'] = self.ravlt_arr_training[:, 0, y_timepoint_idx] # 0 for baseline, only use baseline score
        # self.y_dct['BL_RAVLT30']['train'] = self.ravlt_arr_training[:, 1, y_timepoint_idx]
        # self.y_dct['BL_RAVLT30_RECOG']['train'] = self.ravlt_arr_training[:, 2, y_timepoint_idx]
        # self.y_dct['BL_RAVLT_TOTAL']['test'] = self.ravlt_arr_test[:, 0, y_timepoint_idx]
        # self.y_dct['BL_RAVLT30']['test'] = self.ravlt_arr_test[:, 1, y_timepoint_idx]
        # self.y_dct['BL_RAVLT30_RECOG']['test'] = self.ravlt_arr_test[:, 2, y_timepoint_idx]
        
        # for modality in range(self.modality):
        #     for target_label in self.target_labels:
        #         self.y_dct[target_label]["train"] = self.target_arr_training[:, modality, y_timepoint_idx]
        # for modality in range(self.modality):
        #     for target_label in self.target_labels:
        #         self.y_dct[target_label]["test"] = self.target_arr_test[:, modality, y_timepoint_idx]

        for target_label, modality in zip(self.target_labels, range(self.modality)):
            self.y_dct[target_label]["train"] = self.target_arr_training[:, modality, y_timepoint_idx]
        for target_label, modality in zip(self.target_labels, range(self.modality)):
            self.y_dct[target_label]["test"] = self.target_arr_test[:, modality, y_timepoint_idx]

        #%% Set x for raw(original)
        self.x_dct = {'raw':{}, 'enriched': {}}
        self.x_dct['raw']['train'] = self.X_all[:self.num_of_training_set, :, x_timepoint_idx]
        self.x_dct['raw']['test'] = self.X_all[self.num_of_training_set:, :, x_timepoint_idx]
        #%% Set x for enriched
        self.x_dct['enriched']['train'] = []
        for i in range(self.num_of_training_set):
            self.x_dct['enriched']['train'].append(self.W[i, :].transpose() @ self.X_all[i, :, x_timepoint_idx])
        self.x_dct['enriched']['train'] = np.stack(self.x_dct['enriched']['train'], axis = 0)
        self.x_dct['enriched']['test'] = []
        for i in range(self.num_of_training_set, self.num_of_patients):
            self.x_dct['enriched']['test'].append(self.W[i, :].transpose() @ self.X_all[i, :, x_timepoint_idx])
        self.x_dct['enriched']['test'] = np.stack(self.x_dct['enriched']['test'], axis = 0)
        if min_max_enriched:
            if False:
                min_train = np.min(self.x_dct['enriched']['train'])
                max_train = np.max(self.x_dct['enriched']['train'])
                min_test = np.min(self.x_dct['enriched']['test'])
                max_test = np.max(self.x_dct['enriched']['test'])
                min_ = min(min_train, min_test)
                max_ = max(max_test, max_train)
                self.x_dct['enriched']['train'] = helpers.min_max_scale(self.x_dct['enriched']['train'], min_, max_)
                self.x_dct['enriched']['test'] = helpers.min_max_scale(self.x_dct['enriched']['test'], min_, max_)
            else:
                for fi in range(self.x_dct['enriched']['train'].shape[1]):
                    min_train = np.min(self.x_dct['enriched']['train'][:, fi])
                    max_train = np.max(self.x_dct['enriched']['train'][:, fi])
                    min_test = np.min(self.x_dct['enriched']['test'][:, fi])
                    max_test = np.max(self.x_dct['enriched']['test'][:, fi])
                    min_ = min(min_train, min_test)
                    max_ = max(max_test, max_train)
                    self.x_dct['enriched']['train'][:, fi] = helpers.min_max_scale(self.x_dct['enriched']['train'][:, fi], min_, max_)
                    self.x_dct['enriched']['test'][:, fi] = helpers.min_max_scale(self.x_dct['enriched']['test'][:, fi], min_, max_)
        
        self.img_paths_data = {"enriched": {}, "raw": {}}
        for form in ["enriched", "raw"]:
             self.img_paths_data[form]["train"] = self.img_paths[:self.num_of_training_set]
             self.img_paths_data[form]["test"] = self.img_paths[self.num_of_training_set:]
        
        # self.ravlt_labels_readable = {}
        # self.ravlt_labels_readable['BL_RAVLT_TOTAL'] = 'RAVLT TOTAL'
        # self.ravlt_labels_readable['BL_RAVLT30'] = 'RAVLT 30'
        # self.ravlt_labels_readable['BL_RAVLT30_RECOG'] = 'RAVLT RECOG'
        self.target_labels_readable_dict = copy.deepcopy(self.data_info_dct["target_labels_readable_dict"])
        self.total_RMSE_dct = {}
        for label in self.target_labels:
            self.total_RMSE_dct[self.target_labels_readable_dict[label]] = {'RMSE':{'Original Representation': [], 'Enriched Representation': []}, 'std':{'Original Representation': [], 'Enriched Representation': []}, 'models': []}
        print('init done')

    def lasso_predict(self, grid_param_dict = {'alpha':[100., 10., 1.0, 0.1, 0.01, 0.001, 0.001]}): # Large alpha gives overfitted result(all output same for different inputs),  alpha = 0.001
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                # clf = linear_model.Lasso(alpha = alpha)
                clf = GridSearchCV(linear_model.Lasso(), grid_param_dict, cv = 5, scoring= rmse_sklearn)
                clf.fit(self.x_dct[x_form]['train'], self.y_dct[label]['train'])
                prediction_dct[label][x_form] = clf.predict(self.x_dct[x_form]['test'])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(clf.cv_results_['std_test_score'][clf.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('Lasso')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct

    def SVR_predict(self, grid_param_dict = {'C': [1.0], 'epsilon': [0.1], 'kernel':['rbf', 'poly']}): # C = 0.01, epsilon = 0.1, ['linear', 'poly', 'rbf', 'sigmoid']
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                # clf = svm.SVR(C = C, epsilon = epsilon)
                clf = GridSearchCV(svm.SVR(), grid_param_dict, cv = 5, scoring= rmse_sklearn)
                clf.fit(self.x_dct[x_form]['train'], self.y_dct[label]['train'])
                prediction_dct[label][x_form] = clf.predict(self.x_dct[x_form]['test'])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(clf.cv_results_['std_test_score'][clf.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('SVR')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct
    
    def ridge_predict(self, grid_param_dict = {'alpha':[100., 10., 1.0, 0.1, 0.01, 0.001]}): # alpha = 1.0
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                # clf = linear_model.Ridge(alpha = alpha)
                clf = GridSearchCV(linear_model.Ridge(), grid_param_dict, cv = 5, scoring= rmse_sklearn)
                clf.fit(self.x_dct[x_form]['train'], self.y_dct[label]['train'])
                prediction_dct[label][x_form] = clf.predict(self.x_dct[x_form]['test'])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(clf.cv_results_['std_test_score'][clf.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('RR')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct
    
    def linearRegression_predict(self, grid_param_dict = {'normalize':[True,False]}):
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                # clf = linear_model.LinearRegression()
                clf = GridSearchCV(linear_model.LinearRegression(), grid_param_dict, cv = 5, scoring= rmse_sklearn)
                clf.fit(self.x_dct[x_form]['train'], self.y_dct[label]['train'])
                prediction_dct[label][x_form] = clf.predict(self.x_dct[x_form]['test'])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(clf.cv_results_['std_test_score'][clf.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('LR')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct
    
    def CNN_predict(self, grid_param_dict):
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                #%% CNN structure
                model = KerasRegressor(build_fn=self.create_CNN_model, epochs=500, batch_size=16, verbose=0)
                grid_param_dict_ = copy.deepcopy(grid_param_dict)
                grid_param_dict_['x_form'] = [x_form]
                grid = GridSearchCV(estimator=model, param_grid=grid_param_dict_, scoring= rmse_sklearn, cv = 5)
                # model.summary()

                #%% fit CNN
                grid.fit(self.x_dct[x_form]['train'].reshape((self.num_of_training_set, self.x_length[x_form], 1)), self.y_dct[label]['train'])
                prediction_dct[label][x_form] = (grid.predict(self.x_dct[x_form]['test'].reshape((self.num_of_patients - self.num_of_training_set, self.x_length[x_form], 1)))).reshape(self.num_of_patients - self.num_of_training_set)
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(grid.cv_results_['std_test_score'][grid.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('CNN')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct
    
    def DNN_predict(self, grid_param_dict = {}):
        #%% Prediction
        prediction_dct = {}
        for label in self.target_labels:
            prediction_dct[label] = {}
        for x_form in self.raw_enriched:
            for label in self.target_labels:
                #%% DNN structure
                model = KerasRegressor(build_fn=self.create_DNN_model, epochs=500, batch_size=16, verbose=0)
                grid_param_dict_ = copy.deepcopy(grid_param_dict)
                grid_param_dict_['x_form'] = [x_form]
                grid = GridSearchCV(estimator=model, param_grid=grid_param_dict_, scoring= rmse_sklearn, cv = 5)
                # model.summary()

                #%% fit DNN
                grid.fit(self.x_dct[x_form]['train'], self.y_dct[label]['train'])
                prediction_dct[label][x_form] = (grid.predict(self.x_dct[x_form]['test'])).reshape(self.num_of_patients - self.num_of_training_set)
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['std'][self.readable_raw_enriched[x_form]].append(grid.cv_results_['std_test_score'][grid.best_index_])
        #%% Get RMSE dict
        RMSE_dct = {}
        for label in self.target_labels:
            RMSE_dct[label] = {}
            self.total_RMSE_dct[self.target_labels_readable_dict[label]]['models'].append('DNN')
            for x_form in self.raw_enriched:
                RMSE_dct[label][x_form] = helpers.get_RMSE(self.y_dct[label]['test'], prediction_dct[label][x_form], features_range_dict= self.features_range_dict[label])
                self.total_RMSE_dct[self.target_labels_readable_dict[label]]['RMSE'][self.readable_raw_enriched[x_form]].append(RMSE_dct[label][x_form])
        #%% collects parameter info
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        param_dct = {}
        for arg in args:
            if arg != 'self':
                param_dct[arg] = values[arg]
        return prediction_dct, RMSE_dct, param_dct
    
    def create_CNN_model(self, filters_factor, kernel_size_factor, neurons_factor, drop_rate, x_form):
        model = Sequential()
        model.add(Conv1D(filters = int(filters_factor * 16), kernel_size = int(kernel_size_factor*5), activation = 'relu', input_shape = (self.x_length[x_form], 1), padding = 'valid'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Conv1D(filters = int(filters_factor * 32), kernel_size = int(kernel_size_factor*10), activation = 'relu', padding = 'valid'))
        model.add(MaxPool1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(int(neurons_factor*120)))
        model.add(Dropout(rate = drop_rate))
        model.add(Dense(int(neurons_factor*60)))
        model.add(Dropout(rate = drop_rate))
        model.add(Dense(int(neurons_factor*20)))
        model.add(Dropout(rate = drop_rate))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = utilsforminds.math.get_RMSE_keras)
        return model

    def create_DNN_model(self, x_form, activation = 'relu', neurons_factor = 1.0, drop_rate_factor = 1.0):
        input_length = self.x_length[x_form]
        model = Sequential()
        model.add(Dense(int(input_length * neurons_factor // 1.5), input_dim=input_length, activation='relu'))
        model.add(Dense(int(input_length * neurons_factor // 3), activation='relu'))
        model.add(Dense(int(input_length * neurons_factor // 6), activation='relu'))
        model.add(Dense(int(input_length * neurons_factor // 12), activation='relu'))
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer = 'adam', loss = utilsforminds.math.get_RMSE_keras)
        return model

    def save_result(self, dir_to_save, suffix = '', replace_result_dict = None):
        if replace_result_dict is None:
            replace_result_dict = self.total_RMSE_dct
        for label, result_dict in replace_result_dict.items():
            name_errors = None if 'std' not in replace_result_dict[label].keys() else replace_result_dict[label]['std']
            helpers.plot_bar_charts(dir_to_save + f'/{label.replace(" ", "_")}_{suffix}.eps', result_dict['RMSE'], result_dict['models'], ytitle=f'RMSE of Prediction of {label}', name_not_to_show_percentage_legend = "Original Representation", name_errors = name_errors, name_to_show_percentage = "Enriched Representation")
# class CNN_model():
#     def __init__(self, )