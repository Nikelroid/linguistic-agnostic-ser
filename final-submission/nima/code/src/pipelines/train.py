import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

class LayerProber:
    """
    Implements regression/classification probes to predict acoustic features 
    or emotion labels from hidden layer representations.
    """
    def __init__(self, task_type='regression', alpha=1.0, max_iter=2000):
        self.task_type = task_type
        if task_type == 'regression':
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LogisticRegression(max_iter=max_iter, solver='lbfgs', C=1.0)
            
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if task_type == 'regression' else None
        
    def evaluate(self, X, y, n_splits=5, random_state=42, verbose=False):
        """
        Evaluates the probe using K-Fold cross validation.
        X: array-like of shape (n_samples, hidden_size)
        y: array-like of shape (n_samples,)
        Returns evaluation metric depending on task type.
        """
        if self.task_type == 'regression':
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = []
        else:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            acc_scores, f1_scores = [], []
            
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train = self.scaler_X.fit_transform(X_train)
            X_test = self.scaler_X.transform(X_test)
            
            if self.task_type == 'regression':
                y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                
                self.model.fit(X_train, y_train_scaled)
                preds_scaled = self.model.predict(X_test)
                
                preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                scores.append(rmse)
            else:
                self.model.fit(X_train, y_train)
                preds = self.model.predict(X_test)
                acc_scores.append(accuracy_score(y_test, preds))
                f1_scores.append(f1_score(y_test, preds, average='weighted'))
                
        if self.task_type == 'regression':
            return np.mean(scores)
        else:
            return np.mean(acc_scores), np.std(acc_scores), np.mean(f1_scores)

def probe_all_layers(hidden_states, labels, n_splits=5, random_state=42, task_type='classification'):
    """
    Train a classifier/regressor on each layer's hidden states to predict labels using K-Fold CV.
    hidden_states: numpy array of shape (num_samples, num_layers, hidden_dim)
    """
    num_layers = hidden_states.shape[1]
    y = np.array(labels)
    prober = LayerProber(task_type=task_type)
    results = []
    
    print(f'Probing {num_layers} layers with {task_type}...')
    
    for layer_idx in range(num_layers):
        X = hidden_states[:, layer_idx, :]
        
        if task_type == 'regression':
            rmse = prober.evaluate(X, y, n_splits, random_state)
            layer_name = 'CNN' if layer_idx == 0 else f'Layer {layer_idx}'
            results.append({
                'Layer': layer_name,
                'Layer_Num': layer_idx,
                'RMSE': round(rmse, 4)
            })
            print(f'  {layer_name:10s} | RMSE: {rmse:.4f}')
            
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "layer": layer_idx,
                    "rmse": rmse,
                    "probing_progress_pct": round((layer_idx + 1) / num_layers * 100, 2)
                })
        else:
            mean_acc, std_acc, mean_f1 = prober.evaluate(X, y, n_splits, random_state)
            layer_name = 'CNN' if layer_idx == 0 else f'Layer {layer_idx}'
            results.append({
                'Layer': layer_name,
                'Layer_Num': layer_idx,
                'Accuracy': round(mean_acc, 4),
                'Std_Accuracy': round(std_acc, 4),
                'Weighted_F1': round(mean_f1, 4)
            })
            print(f'  {layer_name:10s} | Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}) | F1: {mean_f1:.4f}')
            
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "layer": layer_idx,
                    "accuracy": mean_acc,
                    "std_accuracy": std_acc,
                    "weighted_f1": mean_f1,
                    "probing_progress_pct": round((layer_idx + 1) / num_layers * 100, 2)
                })
            
    results_df = pd.DataFrame(results)
    if task_type == 'classification':
        best_layer = results_df.loc[results_df['Accuracy'].idxmax()]
        print(f'\\n  BEST LAYER: {best_layer["Layer"]} (Acc={best_layer["Accuracy"]:.4f})')
        
    return results_df
