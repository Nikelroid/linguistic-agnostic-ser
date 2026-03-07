import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

class LayerProber:
    """
    Implements simple regression/classification probes to predict acoustic features 
    from hidden layer representations, as described in the probing paper.
    """
    def __init__(self, task_type='regression', alpha=1.0):
        self.task_type = task_type
        if task_type == 'regression':
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LogisticRegression(max_iter=1000)
            
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if task_type == 'regression' else None
        
    def evaluate(self, X, y, n_splits=5):
        """
        Evaluates the probe using K-Fold cross validation.
        X: array-like of shape (n_samples, hidden_size)
        y: array-like of shape (n_samples,)
        Returns the mean evaluation metric (RMSE for regression, Accuracy for classification).
        """
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train = self.scaler_X.fit_transform(X_train)
            X_test = self.scaler_X.transform(X_test)
            
            if self.task_type == 'regression':
                y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                
                self.model.fit(X_train, y_train_scaled)
                preds_scaled = self.model.predict(X_test)
                
                # Inverse transform predictions to calculate true RMSE
                preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                scores.append(rmse)
            else:
                self.model.fit(X_train, y_train)
                preds = self.model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                scores.append(acc)
                
        return np.mean(scores)
