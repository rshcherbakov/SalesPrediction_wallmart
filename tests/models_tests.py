import unittest
import pandas as pd
from my_lightgbm_wrapper import LightGBMRegressor

# Temp test example


class TestLightGBMRegressor(unittest.TestCase):
    
    def setUp(self):
        """  Generate toy data for testing
        """
        self.X_train = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5],
                                     'feature_2': [2, 4, 6, 8, 10]})
        self.y_train = pd.Series([1, 3, 5, 7, 9])
        self.X_test = pd.DataFrame({'feature_1': [6, 7, 8, 9, 10],
                                    'feature_2': [12, 14, 16, 18, 20]})
        self.y_test = pd.Series([11, 13, 15, 17, 19])
    

    def test_hyperparameter_optimization(self):
        """  Test that hyperparameter optimization returns a dictionary
        """
        lgbm = LightGBMRegressor()
        optuna_params = lgbm.optimize_hyperparameters(self.X_train, self.y_train)
        self.assertIsInstance(optuna_params, dict)
    

    def test_train(self):
        """ Test that the model can be trained with best hyperparameters
        """
        lgbm = LightGBMRegressor()
        lgbm.optimize_hyperparameters(self.X_train, self.y_train)
        lgbm.train(self.X_train, self.y_train)
        self.assertIsNotNone(lgbm.model)
    

    def test_predict(self):
        """ Test that the model can make predictions
        """
        lgbm = LightGBMRegressor()
        lgbm.optimize_hyperparameters(self.X_train, self.y_train)
        lgbm.train(self.X_train, self.y_train)
        y_pred = lgbm.predict(self.X_test)
        self.assertIsInstance(y_pred, pd.Series)
    

    def test_validation(self):
        """ Test that the model validation metrics are calculated correctly
        """
        lgbm = LightGBMRegressor()
        lgbm.optimize_hyperparameters(self.X_train, self.y_train)
        lgbm.train(self.X_train, self.y_train)
        mae, rmse = lgbm.validation(self.X_test, self.y_test)
        self.assertIsInstance(mae, float)
        self.assertIsInstance(rmse, float)