import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load_data(self):
        # Define column names for the DataFrame
        columns = []
        columns.append('System ID')
        columns.append('Data Type')
        columns.append('Class') 
        #column_names = ['System ID', 'Data Type', 'Class']
        # Read the CSV file into a DataFrame, selecting specific columns
        data = pd.read_csv(self.file_path, header=None, usecols=[0, 1, 2], names=columns)
        
        # Read the remaining columns directly into a list of lists
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            # Split each line and skip the first three columns
            data_strings = []
            # Iterate through each line in the file
            for line in lines:
                # Split the line by commas and select elements starting from the fourth element
                split_line = line.strip().split(',')
                selected_elements = split_line[3:]
                # Join the selected elements with commas and add to the data_strings list
                concatenated_string = ','.join(selected_elements)
                data_strings.append(concatenated_string)

        # Add the concatenated data strings as a new 'Data' column to the DataFrame
        data['Data'] = data_strings

        return data
   
    def datafiltering(self, data, data_type):
        # Mapping of abbreviated data types to their corresponding full names
        data_type_mapping = {
            'dia': 'BP Dia_mmHg',
            'sys': 'LA Systolic BP_mmHg',
            'eda': 'EDA_microsiemens',
            'res': 'Respiration Rate_BPM'
        }
        
        # Check if a specific data type is requested and filter accordingly
        if data_type != 'all':
            full_data_type = data_type_mapping.get(data_type)
            if full_data_type:
                return data[data['Data Type'] == full_data_type]
        return data

class FeatureExtractor:
    def __init__(self):
        pass

    def featureextration(self, data):
    # Create a copy of the DataFrame
        data_copy = data.copy()
        # Split the 'Data' column values by comma, convert to float array, and assign back to 'Data' column
        data_copy['Data'] = data_copy['Data'].apply(lambda x: np.fromstring(x, dtype=float, sep=','))
        # Calculate mean, variance, min, and max for each array in 'Data' column
        features = data_copy['Data'].apply(lambda x: [np.mean(x), np.var(x), np.min(x), np.max(x)])
        # Create a DataFrame from the features list and assign column names
        features_df = pd.DataFrame(features.tolist(), columns=['mean', 'variance', 'min', 'max'])
        return features_df

class DataPlotter:
    def __init__(self, data):
        self.data = data

    def plot_instance(self, system_id):
        instance_data = self.data[self.data['System ID'] == system_id]
        data_types = instance_data['Data Type'].unique()

        # Define a color map for data types
        cmap = plt.get_cmap('tab10')  # You can use different colormaps if needed
        colors = cmap(np.linspace(0, 1, len(data_types)))  # Generate colors for data types

        plt.figure(figsize=(10, 6))
        for i, data_type in enumerate(data_types):
            data_type_data = instance_data[(instance_data['Data Type'] == data_type) & (instance_data['Class'] == 'Pain')]
            for index, row in data_type_data.iterrows():
                data_values = [float(val) for val in row['Data'].split(',')]
                plt.plot(data_values, label=f"{row['Data Type']}", color=colors[i])

        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(f'Physiological Data for Instance {system_id}')
        plt.legend()
        plt.grid(True)
        plt.show()
class ModelTrainer:
    def __init__(self):
        pass

    def modeltraining(self, features, labels):
        # Initialize a Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        # Initialize KFold cross-validation with 10 splits
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        performance_metrics = []

        # Iterate through each fold
        for train_index, test_index in kf.split(features):
            # Train the model on the training data
            model.fit(features.iloc[train_index], labels.iloc[train_index])
            # Make predictions on the test data
            predictions = model.predict(features.iloc[test_index])

            # Calculate confusion matrix and performance metrics for the fold
            cm = confusion_matrix(labels.iloc[test_index], predictions, labels=model.classes_)
            accuracy, precision, recall = self.calculate_performance_metrics(labels.iloc[test_index], predictions)

            performance_metrics.append((cm, accuracy, precision, recall))

        # Calculate average performance metrics across all folds
        return self.calculate_average_performance_metrics(performance_metrics)

    def calculate_performance_metrics(self, y_true, y_pred):
        # Calculate accuracy, precision, and recall
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='Pain')
        recall = recall_score(y_true, y_pred, pos_label='Pain')
        return accuracy, precision, recall

    def calculate_average_performance_metrics(self, performance_metrics):
        # Calculate average confusion matrix and performance metrics across all folds
        cm_avg = np.mean([metric[0] for metric in performance_metrics], axis=0)
        acc_avg = np.mean([metric[1] for metric in performance_metrics])
        prec_avg = np.mean([metric[2] for metric in performance_metrics])
        recall_avg = np.mean([metric[3] for metric in performance_metrics])
        return cm_avg, acc_avg, prec_avg, recall_avg

class Main:
    def __init__(self, file_path, data_type):
        self.file_path = file_path
        self.data_type = data_type

    def run(self):
        # Initialize DataLoader with file path
        data_loader = DataLoader(self.file_path)
        # Load data from file
        data = data_loader.load_data()
        #print(data)
        # Filter data based on specified data type
        filtered_data = data_loader.datafiltering(data, self.data_type)
        # Initialize FeatureExtractor
        feature_extractor = FeatureExtractor()
        # Extract features from filtered data
        features = feature_extractor.featureextration(filtered_data)
        # Get labels from filtered data
        labels = filtered_data['Class']

        print("Number of samples:", len(features))  # Print the number of samples
        # Initialize ModelTrainer
        model_trainer = ModelTrainer()
        # Train the model and get performance metrics
        cm, accuracy, precision, recall = model_trainer.modeltraining(features, labels)

        print(f"Confusion Matrix:\n{cm}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        features.boxplot()
        plt.show()
        plotter = DataPlotter(data)
        plotter.plot_instance('F001')

# Entry point
if __name__ == "__main__":
    import sys
    # Get data type and file path from command line arguments
    datatype = sys.argv[1]
    filepath = sys.argv[2]
    # Initialize Main with file path and data type
    main = Main(filepath, datatype)
    main.run()
