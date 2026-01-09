import os
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score

# Ensure output directory exists
os.makedirs('models', exist_ok=True)

# Load Data
fname_treat = '../preprocess_data/PI_treatments_dict.p'
with open(fname_treat, 'rb') as dbfile:
    PI_treatments = pickle.load(dbfile)

fname_res = '../preprocess_data/PI_drug_res_dict.p'
with open(fname_res, 'rb') as dbfile:
    PR_drug_res = pickle.load(dbfile)

for drug in PR_drug_res:
    low_res_seqs = PR_drug_res[drug]['low_res']
    high_res_seqs = PR_drug_res[drug]['high_res']
    
    # Target vector: 0 for low resistance, 1 for high resistance
    targets = np.array([0] * low_res_seqs.shape[0] + [1] * high_res_seqs.shape[0])
    drug_res_seqs = np.vstack([low_res_seqs, high_res_seqs])

    for treatment in PI_treatments:
        print(f"\n--- Drug: {drug} | Treatment: {treatment} ---")
        
        # Fit encoder on the specific combination (includes treatment sequences)
        all_seqs = np.vstack([drug_res_seqs, PI_treatments[treatment]])
        encoder = OneHotEncoder(handle_unknown='ignore')
        trained_encoder = encoder.fit(all_seqs)
        
        # Transform the drug resistance training data
        X = trained_encoder.transform(drug_res_seqs)
        y = targets

        # 5-Fold Stratified CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        # Hyperparameter search for C
        param_grid = {'C': [0.1, 1.0, 10.0]}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # GridSearch within the fold
            grid = GridSearchCV(
                LogisticRegression(penalty='l2', max_iter=10000, class_weight='balanced'),
                param_grid,
                scoring='f1_weighted',
                cv=3
            )
            grid.fit(X_train, y_train)
            
            # Validation
            best_fold_model = grid.best_estimator_
            y_pred = best_fold_model.predict(X_test)
            score = f1_score(y_test, y_pred, average='weighted')
            fold_metrics.append(score)
            
            print(f"  Fold {fold}: Best C={grid.best_params_['C']}, F1={score:.4f}")

        # Final fit on the full drug resistance set for this treatment
        final_search = GridSearchCV(
            LogisticRegression(penalty='l2', max_iter=10000, class_weight='balanced'),
            param_grid,
            scoring='f1_weighted',
            cv=5
        )
        final_search.fit(X, y)
        
        # Save model, encoder, and CV statistics
        info = {
            'onehot_encoder': trained_encoder,
            'logistic_regression': final_search.best_estimator_,
            'cv_f1_mean': np.mean(fold_metrics),
            'cv_f1_std': np.std(fold_metrics),
            'best_params': final_search.best_params_,
            'feature_names': trained_encoder.get_feature_names_out() if hasattr(trained_encoder, 'get_feature_names_out') else None
        }

        out_path = f'models/treatment{treatment}_drugres{drug}_onehot_logistic_v2.p'
        with open(out_path, 'wb') as dbfile:
            pickle.dump(info, dbfile)
        
        print(f"DONE: {out_path} | CV Mean F1: {info['cv_f1_mean']:.4f}")





