import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif, SelectFromModel
import wandb

def main(args):
    # Initialize WandB for EXP_ID 10
    exp_id = args.exp_id
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=f"EXP{exp_id}",
        name=f"metadata_probing",
        reinit=True,
        config=vars(args)
    )

    labels_file = os.path.join(args.data_dir, "v1", "labels_consensus.csv")
    if not os.path.exists(labels_file):
        print(f"File not found: {labels_file}")
        return

    print("Loading all MSP-Podcast metadata...")
    df = pd.read_csv(labels_file)

    # Filter out missing target or features
    df = df.dropna(subset=['EmoClass', 'EmoAct', 'EmoVal', 'EmoDom', 'Gender'])

    # Prepare features
    # Map Gender to numerical
    le_gender = LabelEncoder()
    df['Gender_Num'] = le_gender.fit_transform(df['Gender'].astype(str))

    feature_cols = ['EmoAct', 'EmoVal', 'EmoDom', 'Gender_Num']
    X = df[feature_cols].values

    # Map target EmoClass
    le_target = LabelEncoder()
    y = le_target.fit_transform(df['EmoClass'].astype(str))

    print(f"Total samples: {len(y)}")
    print(f"Classes: {le_target.classes_}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Statistical Analysis & Feature Importance ---")
    
    # 1. Mutual Information
    mi = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': feature_cols, 'Mutual_Information': mi})
    mi_df = mi_df.sort_values(by='Mutual_Information', ascending=False)
    print("\nMutual Information:")
    print(mi_df.to_string(index=False))

    # 2. Random Forest Importance
    rf_explainer = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_explainer.fit(X_scaled, y)
    rf_imp = rf_explainer.feature_importances_
    rf_df = pd.DataFrame({'Feature': feature_cols, 'RF_Importance': rf_imp})
    rf_df = rf_df.sort_values(by='RF_Importance', ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(rf_df.to_string(index=False))

    # Log importances to WandB
    wandb.log({
        "Mutual_Information": wandb.Table(dataframe=mi_df),
        "RF_Importance": wandb.Table(dataframe=rf_df)
    })

    # Select features with higher effect
    # We will use SelectFromModel with Random Forest, which picks features with importance >= mean
    selector = SelectFromModel(rf_explainer, prefit=True, threshold="mean")
    X_selected = selector.transform(X_scaled)
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
    
    # If it filters out too much (e.g. only 1 feature left), let's ensure we keep at least top 2
    if len(selected_features) < 2:
        top_2_idx = np.argsort(rf_imp)[::-1][:2]
        X_selected = X_scaled[:, top_2_idx]
        selected_features = [feature_cols[i] for i in top_2_idx]

    print(f"\nSelected Features for Modeling: {selected_features}")
    wandb.config.update({"selected_features": selected_features})

    print("\n--- Model Evaluation (5-Fold Cross Validation) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP_Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        # Accuracy
        acc_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)
        # Macro F1
        f1_scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1_macro', n_jobs=-1)
        
        mean_acc = np.mean(acc_scores)
        mean_f1 = np.mean(f1_scores)
        
        print(f"  Accuracy: {mean_acc:.4f} (+/- {np.std(acc_scores):.4f})")
        print(f"  Macro F1: {mean_f1:.4f} (+/- {np.std(f1_scores):.4f})")
        
        results.append({
            "Model": name,
            "Accuracy": mean_acc,
            "Macro_F1": mean_f1
        })
        
        wandb.log({
            f"{name}_Accuracy": mean_acc,
            f"{name}_Macro_F1": mean_f1
        })

    results_df = pd.DataFrame(results)
    
    # Save results
    results_base = os.path.join("results", f"EXP{exp_id}")
    os.makedirs(results_base, exist_ok=True)
    out_path = os.path.join(results_base, "metadata_probing_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw dataset")
    parser.add_argument("--exp_id", type=str, default="9")
    parser.add_argument("--wandb_entity", type=str, default="AGSER")
    parser.add_argument("--wandb_project", type=str, default="linguistic-agnostic-ser")
    args = parser.parse_args()
    main(args)
