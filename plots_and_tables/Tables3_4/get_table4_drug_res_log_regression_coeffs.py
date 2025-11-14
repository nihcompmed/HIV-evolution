import os
import pickle
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

# --- Configuration ---
MODEL_DIR = '/Users/aggarwalm4/Downloads/HIV_revision/FINAL_REVISION_FOLDER/logistic_epistatic_model/logistic_training'
OUTPUT_DIR = 'resistance_feature_summary'
OUTPUT_FILENAME = 'summary_top5_features_None_treatment.csv'
TOP_N = 5

def reformat_feature_name(feature_name):
    """
    Converts a feature name from 'x49_V' format to '50V'.
    """
    if feature_name == 'N/A':
        return 'N/A'
    try:
        # Split 'x49_V' into 'x49' and 'V'
        parts = feature_name.split('_')
        # Get the number, remove 'x', convert to int, add 1
        position = int(parts[0][1:]) + 1
        # Get the amino acid
        amino_acid = parts[1]
        return f"{position}{amino_acid}"
    except (IndexError, ValueError):
        # If the format is unexpected, return the original name
        return feature_name

def get_top_positive_features(model_path, top_n=5):
    """
    Loads a pickled model and returns the top N features with positive coefficients,
    reformatted for 1-based indexing.
    """
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"  - Could not load or read file: {os.path.basename(model_path)}. Error: {e}")
        return ['N/A'] * top_n

    onehot_encoder = data.get('onehot_encoder')
    logistic_regression = data.get('logistic_regression')

    if not all([onehot_encoder, logistic_regression]):
        print(f"  - File is missing 'onehot_encoder' or 'logistic_regression': {os.path.basename(model_path)}")
        return ['N/A'] * top_n

    feature_names = onehot_encoder.get_feature_names_out()
    coefficients = logistic_regression.coef_.flatten()

    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    positive_coef_df = coef_df[coef_df['Coefficient'] > 0].copy()
    sorted_positive = positive_coef_df.sort_values(by='Coefficient', ascending=False)
    
    top_features_raw = sorted_positive['Feature'].head(top_n).tolist()
    
    # Reformat the feature names
    top_features = [reformat_feature_name(f) for f in top_features_raw]

    # Pad with 'N/A' if there are fewer than N positive features
    while len(top_features) < top_n:
        top_features.append('N/A')
        
    return top_features

def main():
    """
    Scans for 'None' treatment models and creates a single summary CSV
    where columns are PIs and rows are top resistance features.
    """
    print("="*80)
    print("Creating Summary of Top Resistance Features for 'None' Treatment")
    print("="*80)

    if not os.path.isdir(MODEL_DIR):
        print(f"ERROR: Model directory not found at '{MODEL_DIR}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output will be saved in '{OUTPUT_DIR}/'")

    # --- 1. Scan files and find models for 'None' treatment ---
    none_treatment_models = []
    pattern = re.compile(r"treatmentNone_drugres(.*)_onehot_logistic.p")

    print("\nScanning for 'None' treatment models...")
    for filename in os.listdir(MODEL_DIR):
        match = pattern.match(filename)
        if match:
            pi_name = match.group(1)
            full_path = os.path.join(MODEL_DIR, filename)
            none_treatment_models.append({
                'pi': pi_name,
                'path': full_path
            })
            print(f"- Found model for PI: {pi_name}")

    if not none_treatment_models:
        print("\nNo models found for the 'None' treatment. Exiting.")
        return

    # --- 2. Aggregate results for the 'None' treatment ---
    results_for_none = {}
    
    print("\nProcessing models...")
    for model_info in none_treatment_models:
        pi = model_info['pi']
        print(f"- Analyzing resistance features for PI '{pi}'...")
        top_features = get_top_positive_features(model_info['path'], top_n=TOP_N)
        results_for_none[pi] = top_features

    # --- 3. Create and save the single summary DataFrame ---
    # Sort the dictionary by PI name for consistent column order
    sorted_results = dict(sorted(results_for_none.items()))
    
    summary_df = pd.DataFrame(sorted_results)
    # The DataFrame index is temporary and will not be saved to the file.
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    # Save without the index column
    summary_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully created summary file: {output_path}")

    print("\n" + "="*80)
    print("Summary generation complete.")
    print("="*80)

if __name__ == '__main__':
    main()


