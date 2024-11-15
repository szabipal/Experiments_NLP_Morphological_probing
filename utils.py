import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import re  


def load_data(layer_folder):
    """ Load training and evaluation data for a given layer folder """
    train_path = f"{layer_folder}/train_layer_{layer_folder.split('_')[-1]}.csv"
    eval_path = f"{layer_folder}/eval_layer_{layer_folder.split('_')[-1]}.csv"
    
    train_data = pd.read_csv(train_path)
    eval_data = pd.read_csv(eval_path)
    
    return train_data, eval_data

def prepare_features_and_labels(data, layer_column):
    """ Prepare features (X) and labels (y) from the dataset """
    X = data[layer_column].apply(eval).tolist()  # Convert string representations of lists back to numerical lists
    y = data['Gold_Label']
    return X, y







def prepare_training_and_evaluation_sets(target_representations, non_target_representations, num_non_target_markers):
    """
    Prepares a balanced training and evaluation set with 30% target markers and 70% non-target markers.
    
    Args:
        target_representations (list): Representations of the target marker.
        non_target_representations (list): Representations of non-target markers.
        num_non_target_markers (int): Number of distinct non-target markers.
    
    Returns:
        tuple: Balanced training and evaluation sets as DataFrames.
    """
    # Convert lists to DataFrames for easier manipulation
    target_df = pd.DataFrame(target_representations)
    non_target_df = pd.DataFrame(non_target_representations)
    
    # Add the gold label for target class (1 for target class)
    target_df['Gold_Label'] = 1
    
    # Add the gold label for non-target class (0 for non-target classes)
    non_target_df['Gold_Label'] = 0

    # Calculate the desired sizes for balanced data
    target_size = int(0.3 * len(non_target_representations) * (1 / 0.7))  # 30% of the total size
    non_target_size = target_size * (70 / 30)  # 70% of the total size
    
    # Step 1: Sample the target instances for the training set
    target_train, target_eval = train_test_split(target_df, test_size=0.1, random_state=42)
    target_train = target_train.sample(n=target_size, random_state=42)

    # Step 2: Sample from the non-target class evenly across all non-target markers
    non_target_sample_per_marker = int(non_target_size / num_non_target_markers)
    non_target_train_samples = []
    
    for _ in range(num_non_target_markers):
        non_target_sample = non_target_df.sample(n=non_target_sample_per_marker, random_state=42)
        non_target_train_samples.append(non_target_sample)
    
    non_target_train = pd.concat(non_target_train_samples)
    
    # Step 3: Extract 10% of both the target and non-target instances for evaluation
    non_target_train, non_target_eval = train_test_split(non_target_train, test_size=0.1, random_state=42)

    # Step 4: Combine the remaining target and non-target instances for training
    train_set = pd.concat([target_train, non_target_train])
    eval_set = pd.concat([target_eval, non_target_eval])
    
    # Step 5: Shuffle both the training and evaluation sets
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    eval_set = eval_set.sample(frac=1).reset_index(drop=True)
    
    return train_set, eval_set





def load_files_from_folder(folder_path, file_extension='.pkl'):
    """Loads all files with a specific extension from a given folder."""
    
    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            file_path = os.path.join(folder_path, file)
            if file_extension == '.pkl':
                # If the file is a pickle file, load it
                with open(file_path, 'rb') as f:
                    file=(pickle.load(f))
                    df = pd.DataFrame(file)
            elif file_extension == '.csv':
                # If the file is a CSV file, load it with pandas
                df=pd.read_csv(file_path)
    return df






def separate_target_and_non_target(representations, target_marker):
    """
    Separates the target marker representations and non-target marker representations.
    
    Args:
        representations (dict): Dictionary containing representations organized by marker.
        target_marker (str): The key representing the target marker.
    
    Returns:
        target_reps (pd.DataFrame): Representations corresponding to the target marker.
        non_target_reps (pd.DataFrame): Representations corresponding to all non-target markers.
    """
    # Get the target marker's representations
    target_reps = representations[target_marker]
    non_target_reps=[]
    for key, value in representations.items(): 
        if key != target_marker:
            non_target_reps.append(value)
        
    return target_reps, non_target_reps






def balance_and_create_layered_data(target_reps, non_target_reps_list, marker_name, layered_data_folder='layered_data', target_threshold=200, eval_split=0.15):
    """
    Balances target and non-target representations, splits them into train and eval sets, and saves layer-wise representations.
    
    Args:
        target_reps (pd.DataFrame): Representations for the target class.
        non_target_reps_list (list of pd.DataFrame): List of DataFrames, each containing non-target representations.
        marker_name (str): The name of the marker for folder structure.
        layered_data_folder (str): The base folder for saving layered representations.
        target_threshold (int): Minimum number of target samples. If below, oversample to match this number.
        eval_split (float): Proportion of data to use for evaluation (default is 15%).
    
    Returns:
        None
    """
    


    # Step 1: Oversample target representations if they are fewer than target_threshold
    if len(target_reps) < target_threshold:
        target_reps = resample(target_reps, replace=True, n_samples=target_threshold, random_state=42)
    else:
        target_reps = target_reps.sample(n=target_threshold, random_state=42)

    # Step 2: Divide the number of target instances by 7 to determine how many non-target instances to sample from each label
    non_target_samples_per_label = len(target_reps) // 7
    
    balanced_non_target_reps = []
    
    # Step 3: For each non-target label, oversample or undersample to reach the calculated number
    for non_target_reps in non_target_reps_list:
        if len(non_target_reps) < non_target_samples_per_label:
            # If there are fewer non-target instances than required, oversample
            sampled_non_target_reps = resample(non_target_reps, replace=True, n_samples=non_target_samples_per_label, random_state=42)
        else:
            # Otherwise, sample without replacement
            sampled_non_target_reps = non_target_reps.sample(n=non_target_samples_per_label, random_state=42)
        
        balanced_non_target_reps.append(sampled_non_target_reps)
    
    # Step 4: Combine all the balanced non-target representations into one DataFrame
    balanced_non_target_reps = pd.concat(balanced_non_target_reps, ignore_index=True)
    
    # Step 5: Combine target and non-target representations
    combined_reps = pd.concat([target_reps, balanced_non_target_reps], ignore_index=True)

    # Step 6: Split into train and evaluation sets (85% train, 15% eval)
    train_set, eval_set = train_test_split(combined_reps, test_size=eval_split, random_state=42)
    
    # Step 7: Process hidden representations (12 layers) and save to respective folders
    save_layered_representations(train_set, eval_set, marker_name, layered_data_folder)



def save_layered_representations(train_set, eval_set, marker_name, layered_data_folder):
    """
    Saves the hidden representations for each layer into separate files with the gold label column.
    
    Args:
        train_set (pd.DataFrame): The training set containing hidden representations.
        eval_set (pd.DataFrame): The evaluation set containing hidden representations.
        marker_name (str): The marker name used for folder structure.
        layered_data_folder (str): The base folder where the layered data will be saved.
    
    Returns:
        None
    """
    marker_folder = os.path.join(layered_data_folder, marker_name)
    if not os.path.exists(marker_folder):
        os.makedirs(marker_folder)
    
    num_layers = 12  # Assuming there are 12 layers
    
    for layer_idx in range(num_layers):
        layer_folder = os.path.join(marker_folder, str(layer_idx + 1))
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder)
        
        # Clear any existing files
        train_file_path = os.path.join(layer_folder, "train.csv")
        eval_file_path = os.path.join(layer_folder, "eval.csv")
        
        if os.path.exists(train_file_path):
            os.remove(train_file_path)
        if os.path.exists(eval_file_path):
            os.remove(eval_file_path)
        
        # Prepare the data
        train_layer_data = train_set.copy()
        eval_layer_data = eval_set.copy()

        # Convert each layer's hidden representation into a comma-separated string
        train_layer_data['Representation'] = train_layer_data['Hidden Representations (All Layers)'].apply(
            lambda x: str([float(val) for val in x[layer_idx]]).replace(" ", "")  # Remove spaces between commas
        )
        eval_layer_data['Representation'] = eval_layer_data['Hidden Representations (All Layers)'].apply(
            lambda x: str([float(val) for val in x[layer_idx]]).replace(" ", "")
        )
        
        # Drop the original 'Hidden Representations (All Layers)' column
        train_layer_data = train_layer_data.drop(columns=['Hidden Representations (All Layers)'])
        eval_layer_data = eval_layer_data.drop(columns=['Hidden Representations (All Layers)'])
        
        # Save files
        train_layer_data.to_csv(train_file_path, index=False)
        eval_layer_data.to_csv(eval_file_path, index=False)

        print(f"Layer {layer_idx + 1} representations saved for train and eval datasets in '{layer_folder}'.")




def string_to_list(representation_str):
    """
    Converts a string representation of a list of floats into an actual list of floats.
    
    Args:
        representation_str (str): String representation of the list of floats.
    
    Returns:
        list: List of floats.
    """
    try:
        # Safely evaluate the string as a Python expression
        parsed_list = ast.literal_eval(representation_str)
        # Ensure the result is a list of floats
        return [float(x) for x in parsed_list]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing representation: {e}")
        return []







def run_probing_on_all_layers(layered_data_folder='Layered_data', output_folder='Output_comparison'):
    """
    Loops through the marker folders in Layered_data, extracts the training and eval sets,
    and runs the probing function on them for all layers at once.
    """
    for marker_name in os.listdir(layered_data_folder):
        marker_folder = os.path.join(layered_data_folder, marker_name)
        
        if os.path.isdir(marker_folder):
            print(f"Processing marker: {marker_name}")
            
            # Define paths for the train and eval CSV files
            train_files = [os.path.join(marker_folder, str(layer_idx), 'train.csv') for layer_idx in range(1, 13)]
            eval_files = [os.path.join(marker_folder, str(layer_idx), 'eval.csv') for layer_idx in range(1, 13)]
            
            # Check if all required train and eval files exist
            if all(os.path.exists(train_file) for train_file in train_files) and all(os.path.exists(eval_file) for eval_file in eval_files):
                train_sets = [pd.read_csv(train_file) for train_file in train_files]
                eval_sets = [pd.read_csv(eval_file) for eval_file in eval_files]

                # Run the probing function for all layers at once for the current marker
                train_and_evaluate_logistic_regression(
                    train_sets=train_sets,
                    eval_sets=eval_sets,
                    marker_name=marker_name,
                    output_folder=output_folder
                )
            else:
                print(f"Missing train or eval file(s) for marker: {marker_name}")

                
def train_and_evaluate_logistic_regression(train_sets, eval_sets, marker_name, output_folder='Output_comparison'):
    """
    Trains logistic regression probes across all layers and saves cumulative results.
    """
    output_folder = os.path.join(output_folder, marker_name)
    os.makedirs(output_folder, exist_ok=True)
    
    results_file = os.path.join(output_folder, f'{marker_name}_results.csv')
    cumulative_results_df = pd.DataFrame(columns=['Layer', 'Marker', 'Precision', 'Recall', 'F1-Score'])

    for layer_idx in range(12):  # Process each layer's train and eval sets
        train_set = train_sets[layer_idx]
        eval_set = eval_sets[layer_idx]
        
        # Process the representations for the specific layer
        train_set['Representation'] = train_set['Representation'].apply(lambda x: string_to_list(x)[layer_idx])
        eval_set['Representation'] = eval_set['Representation'].apply(lambda x: string_to_list(x)[layer_idx])
        
        X_train = np.vstack(train_set['Representation'].values)
        X_eval = np.vstack(eval_set['Representation'].values)
        # y_train = train_set[marker_name].values
        # y_eval = eval_set[marker_name].values
        if marker_name == 'Causal_final':    
            y_train = train_set['CausalFinal'].values  # Ensure this is a 1D array
            y_eval = eval_set['CausalFinal'].values
        else:
            y_train = train_set[marker_name].values  # Ensure this is a 1D array
            y_eval = eval_set[marker_name].values

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_eval)

        precision = precision_score(y_eval, y_pred, average='binary')
        recall = recall_score(y_eval, y_pred, average='binary')
        f1 = f1_score(y_eval, y_pred, average='binary')

        # Append results to DataFrame
        cumulative_results_df = pd.concat([cumulative_results_df, pd.DataFrame([{
            'Layer': layer_idx + 1,
            'Marker': marker_name,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }])], ignore_index=True)

    cumulative_results_df.to_csv(results_file, index=False)
    print(f"Cumulative results for all layers of {marker_name} saved to {results_file}")
