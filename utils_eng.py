import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
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

def train_and_evaluate_logistic_regression(train_data, eval_data, layer_idx):
    """ Train and evaluate a logistic regression classifier for a given layer """
    # Prepare features and labels
    X_train, y_train = prepare_features_and_labels(train_data, f'Layer_{layer_idx}_Representation')
    X_eval, y_eval = prepare_features_and_labels(eval_data, f'Layer_{layer_idx}_Representation')

    # Train the logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict on the evaluation set
    y_pred = clf.predict(X_eval)

    # Calculate metrics
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall = recall_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred)

    return accuracy, precision, recall, f1

def compare_layers(marker_name, num_layers):
    """ Compare logistic regression results across layers """
    results = []

    for layer_idx in range(num_layers):
        layer_folder = f"{marker_name}/Layer_{layer_idx}"
        
        # Load training and evaluation data for the current layer
        train_data, eval_data = load_data(layer_folder)
        
        # Train and evaluate the model for this layer
        accuracy, precision, recall, f1 = train_and_evaluate_logistic_regression(train_data, eval_data, layer_idx)

        # Store the results
        results.append({
            'Layer': layer_idx,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    # Convert results to a DataFrame for easy comparison
    results_df = pd.DataFrame(results)
    return results_df





def prepare_training_and_evaluation_sets(target_representations, non_target_representations):
    # Convert lists to DataFrames for easier manipulation
    target_df = pd.DataFrame(target_representations)
    non_target_df = pd.DataFrame(non_target_representations)
    
    # Add the gold label for target class (1 for target class)
    target_df['Gold_Label'] = 1
    
    # Add the gold label for non-target class (0 for non-target classes)
    non_target_df['Gold_Label'] = 0
    
    # Step 1: Extract 10% of the target class instances for evaluation
    target_train, target_eval = train_test_split(target_df, test_size=0.1, random_state=42)
    
    # Step 2: Randomly sample from the non-target class to match the target size
    non_target_sampled = non_target_df.sample(n=len(target_df), random_state=42)
    
    # Step 3: Extract 10% of the non-target instances for evaluation
    non_target_train, non_target_eval = train_test_split(non_target_sampled, test_size=0.1, random_state=42)
    
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
    # Create the base folder for the marker
    marker_folder = os.path.join(layered_data_folder, marker_name)
    if not os.path.exists(marker_folder):
        os.makedirs(marker_folder)
    
    num_layers = 12  # As per BERT, we have 12 layers
    
    for layer_idx in range(num_layers):
        # Create subfolders for each layer inside the marker folder
        layer_folder = os.path.join(marker_folder, str(layer_idx + 1))  # Layer folders are named 1 to 12
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder)
        
        # Define the paths for the train and eval files
        train_file_path = os.path.join(layer_folder, "train.csv")
        eval_file_path = os.path.join(layer_folder, "eval.csv")
        
        # Remove any existing train.csv and eval.csv files in this layer folder to prevent duplication
        if os.path.exists(train_file_path):
            os.remove(train_file_path)
        if os.path.exists(eval_file_path):
            os.remove(eval_file_path)
        
        # Prepare layer-specific data for training and evaluation sets
        train_layer_data = train_set.copy()
        eval_layer_data = eval_set.copy()

        # Extract and add the layer-specific hidden representations as a separate column
        train_layer_data['Representation'] = train_layer_data['Hidden Representations (All Layers)'].apply(lambda x: x[layer_idx])
        eval_layer_data['Representation'] = eval_layer_data['Hidden Representations (All Layers)'].apply(lambda x: x[layer_idx])
        
        # Drop the original 'Hidden Representations (All Layers)' column
        train_layer_data = train_layer_data.drop(columns=['Hidden Representations (All Layers)'])
        eval_layer_data = eval_layer_data.drop(columns=['Hidden Representations (All Layers)'])
        
        # Save the train and eval sets for this layer into CSV files
        train_layer_data.to_csv(train_file_path, index=False)
        eval_layer_data.to_csv(eval_file_path, index=False)

        print(f"Layer {layer_idx + 1} representations saved for train and eval datasets in '{layer_folder}'.")







def string_to_list(representation_str):
    """
    Converts a string representation of a list of lists into an actual list of NumPy arrays.
    
    Args:
        representation_str (str): String representation of the list of lists.
    
    Returns:
        list: List of NumPy arrays.
    """
    # Extract inner lists using regular expressions
    inner_lists = re.findall(r'\[([^\[\]]+)\]', representation_str)
    list_of_arrays = [np.array(list(map(float, inner_list.split()))) for inner_list in inner_lists]
    return list_of_arrays

def train_and_evaluate_logistic_regression(train_set, eval_set, marker_name, output_folder='Output_comparison'):
    """
    Trains logistic regression probes, evaluates them, and saves results in a DataFrame.
    
    Args:
        train_set (pd.DataFrame): The training set containing the 'Representation' and marker gold label columns.
        eval_set (pd.DataFrame): The evaluation set.
        marker_name (str): The marker being probed (target label).
        output_folder (str): The folder where results will be saved.
    
    Returns:
        None
    """
    # Ensure the output folder is correctly set for the marker
    output_folder = os.path.join(output_folder, marker_name)
    
    # Convert 'Representation' from string to list of floats
    X_train = train_set['Representation'].apply(string_to_list).tolist()
    y_train = train_set[marker_name]
    
    X_eval = eval_set['Representation'].apply(string_to_list).tolist()
    y_eval = eval_set[marker_name]

    # Train the logistic regression model
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the evaluation set
    y_pred = clf.predict(X_eval)

    # Calculate evaluation metrics
    precision = precision_score(y_eval, y_pred, average='binary')
    recall = recall_score(y_eval, y_pred, average='binary')
    f1 = f1_score(y_eval, y_pred, average='binary')

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Marker': [marker_name],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })

    # Ensure the marker-specific folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the results file path
    results_file = os.path.join(output_folder, f'{marker_name}_results.csv')

    # Save or append the results to the CSV file
    if os.path.exists(results_file):
        # If the file exists, append without the header
        results_df.to_csv(results_file, mode='a', header=False, index=False)
    else:
        # If the file doesn't exist, create it and write with the header
        results_df.to_csv(results_file, mode='w', header=True, index=False)

    print(f"Results for {marker_name} saved to {results_file}")





def run_probing_on_all_layers(layered_data_folder='Layered_data', output_folder='Output_comparison'):
    """
    Loops through the nested folders in Layered_data, extracts the training and eval sets,
    and runs the probing function on them.
    
    Args:
        layered_data_folder (str): The folder where the layer data is stored.
        output_folder (str): The folder where results will be saved.
    
    Returns:
        None
    """
    # Loop through each marker folder in the Layered_data folder
    for marker_name in os.listdir(layered_data_folder):
        marker_folder = os.path.join(layered_data_folder, marker_name)
        
        # Check if it's a directory (marker folder)
        if os.path.isdir(marker_folder):
            print(f"Processing marker: {marker_name}")
            
            # Loop through each layer folder (1 to 12) inside the marker folder
            for layer_idx in range(1, 13):
                layer_folder = os.path.join(marker_folder, str(layer_idx))
                
                # Define paths for the train and eval CSV files
                train_file = os.path.join(layer_folder, f'train_layer_{layer_idx}.csv')
                eval_file = os.path.join(layer_folder, f'eval_layer_{layer_idx}.csv')
                
                # Check if the train and eval files exist
                if os.path.exists(train_file) and os.path.exists(eval_file):
                    # Load the train and eval sets
                    train_set = pd.read_csv(train_file)
                    eval_set = pd.read_csv(eval_file)
                    
                    # Run the logistic regression probing function for the current marker and layer
                    train_and_evaluate_logistic_regression(
                        train_set=train_set,
                        eval_set=eval_set,
                        marker_name=marker_name,
                        output_folder=output_folder
                    )
                else:
                    print(f"Missing train or eval file for marker: {marker_name}, layer: {layer_idx}")

