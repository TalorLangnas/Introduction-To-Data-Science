from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function for processing the data
# drop rows with missing values
def process_data(df):
    # Handling missing values marked as '?'
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Convert the target variable to a binary numeric variable
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    return df
def process_data_discertization(df: pd.DataFrame):
    # Handling missing values marked as '?'
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    df['income'] = label_encoder.fit_transform(df['income'])
    
    # Discretize the 'age' column into sqrt(74) bins
    num_bins = 8
    # Discretize and replace the 'age' column
    df['age'] = pd.cut(df['age'], bins=num_bins, labels=False)
   
    # Discretize the 'fnlwgt' column into sqrt(26741) bins
    num_bins = 163
    # Discretize and replace the 'fnlwgt' column
    df['fnlwgt'] = pd.cut(df['fnlwgt'], bins=num_bins, labels=False)
   
    # Discretize the 'capital-gain' column into sqrt(121) bins
    num_bins = 11
    # Discretize and replace the 'capital-gain' column
    df['capital-gain'] = pd.cut(df['capital-gain'], bins=num_bins, labels=False)
    
    # Discretize the 'capital-loss' column into sqrt(100) bins
    num_bins = 10
    # Discretize and replace the 'capital-loss' column
    df['capital-loss'] = pd.cut(df['capital-loss'], bins=num_bins, labels=False)
    
    # Discretize the 'hours-per-week' column into sqrt(96) bins
    num_bins = 10
    # Discretize and replace the 'hours-per-week' column
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=num_bins, labels=False)

    return df

class NaiveBayes:
    def __init__(self, class_column: str):
        self.class_column = class_column
        # self.class_value = class_value
        self.word_probability_0 = None
        self.word_probability_1 = None
        self.class_0_probability = None
        self.class_1_probability = None
        self.features_vector = None
        self.class_0_size = None
        self.class_1_size = None

    # Receives the training data, create histogeams and calculate the probabilities
    def fit(self, df_train: pd.DataFrame):
        # get the features vector
        self.features_vector = df.iloc[0].index.to_list()
        self.features_vector.remove('income')

        # divide df_train for dataframe for each class
        df_class_0 = df_train[df_train[self.class_column] == 0]
        df_class_1 = df_train[df_train[self.class_column] == 1]
        # classes size (adding 1 for Laplace smoothing)
        self.class_0_size = df_class_0.shape[0] + 1
        self.class_1_size = df_class_1.shape[0] + 1
        # calculate the word probability for each word in each class
        # add 1 for Laplace smoothing
        self.word_probability_0 = self.get_word_probability(df_class_0, self.class_0_size + 1)
        self.word_probability_1 = self.get_word_probability(df_class_1, self.class_1_size + 1)
        
        # calculate the probability of each class
        self.class_0_probability = self.class_0_size / self.class_0_size + self.class_1_size 
        self.class_1_probability = self.class_1_size / self.class_0_size + self.class_1_size 
    
    # Count the occurrences of each unique value in the DataFrame,
    # calculate the probability and return a dictionary.
    def get_word_probability(self, dataframe: pd.DataFrame, class_size: int):
        
        counts_dict = {}
        # Iterate over each column in the DataFrame
        for column in dataframe.columns:
            # Count occurrences of each unique value in the column
            # Laplace smoothing
            value_counts = (dataframe[column].value_counts()) + 1 / class_size
            # Update the dictionary with counts for each unique value
            counts_dict[column] = value_counts.to_dict()

        return counts_dict

    def get_log_word_probability(self, dataframe: pd.DataFrame, class_size: int):
        
        log_probs_dict = {}

        # Iterate over each column in the DataFrame
        for column in dataframe.columns:
            # Count occurrences of each unique value in the column
            # Laplace smoothing and take the log
            value_counts = np.log((dataframe[column].value_counts() + 1) / class_size)

            # Update the dictionary with log probabilities for each unique value
            log_probs_dict[column] = value_counts.to_dict()

        return log_probs_dict

    # Receives data to be predicted and returns the predicted class
    def predict(self, X_test: np.ndarray):
        predictions = []
        
        for x in X_test:
            predictions.append(self.predict_single(x))
        
        return predictions
    
    # Receives a single data point as 1D np array and returns the predicted class
    def predict_single(self, x):
        if len(x) != len(self.features_vector):
            raise ValueError('The input data point must have the same number of features as the training data')
        
        probability_for_class_0 = 1
        probability_for_class_1 = 1
        # calc the probability that x is in class 0
        # for each word in x
        for i in range(len(self.features_vector)):
            # check for 0 value
            if x[i] not in self.word_probability_0[self.features_vector[i]]:
                probability_for_class_0 *= 1 / self.class_0_size + 1
            else:
                # multiply the probability
                probability_for_class_0 *= self.word_probability_0[self.features_vector[i]][x[i]]
        
        # calc the probability that x is in class 1
        for i in range(len(self.features_vector)):
            # check for 0 value
            if x[i] not in self.word_probability_1[self.features_vector[i]]:
                probability_for_class_1 *= 1 / self.class_1_size + 1
            else:
                # multiply the probability
                probability_for_class_1 *= self.word_probability_1[self.features_vector[i]][x[i]]     
        
        if probability_for_class_0 > probability_for_class_1:
            return 0
        else:    
            return 1
    
    # Receives data to be predicted and returns the predicted class
    def predict_log(self, X_test: np.ndarray):
        predictions = []
        
        for x in X_test:
            predictions.append(self.predict_single_log(x))
        
        return predictions

    # Receives a single data point as 1D np array and returns the predicted class
    def predict_single_log(self, x):
        if len(x) != len(self.features_vector):
            raise ValueError('The input data point must have the same number of features as the training data')
        
        log_probability_for_class_0 = 0
        log_probability_for_class_1 = 0
        # calc the probability that x is in class 0
        # for each word in x
        for i in range(len(self.features_vector)):
            # check for 0 value
            if x[i] not in self.word_probability_0[self.features_vector[i]]:
                log_probability_for_class_0 += np.log(1 / (self.class_0_size + 1))
            else:
                # multiply the probability
                log_probability_for_class_0 += np.log(self.word_probability_1[self.features_vector[i]][x[i]])
        
        # calc the probability that x is in class 1
        for i in range(len(self.features_vector)):
            # check for 0 value
            if x[i] not in self.word_probability_1[self.features_vector[i]]:
                log_probability_for_class_1 += np.log(1 / (self.class_1_size + 1))
            else:
                # multiply the probability
                log_probability_for_class_1 += np.log(self.word_probability_1[self.features_vector[i]][x[i]])   
        
        if log_probability_for_class_0 > log_probability_for_class_1:
            return 0
        else:    
            return 1                 
      
    # Receives the predictions and the actual values and returns the 
    # accuracy, precision, recall, and f-measure
    def evaluate(self, predictions, y_test):
        evaluations = []
        acc = accuracy_score(y_test, predictions)
        print("accuracy: ", acc)
        evaluations.append(acc)
        ps = precision_score(y_test, predictions)
        print("precision: ", ps)
        evaluations.append(ps)
        rs = recall_score(y_test, predictions)
        print("recall: ", rs)
        evaluations.append(rs)
        f1 = f1_score(y_test, predictions)
        print("f1: ", f1)
        evaluations.append(f1)

        # return evaluations
        return acc, ps, rs, f1

def balance_sample(df: pd.DataFrame, n_samples: int):
    # Separate the data into two classes
    class_0 = df[df['income'] == 0]
    class_1 = df[df['income'] == 1]

    # Sample half the size from each class
    n_samples = min(len(class_0), len(class_1))

    # Use sample function to get random samples
    sampled_class_0 = class_0.sample(n=n_samples, random_state=42)
    sampled_class_1 = class_1.sample(n=n_samples, random_state=42)

    # Concatenate the sampled dataframes
    balanced_sample = pd.concat([sampled_class_0, sampled_class_1])

    # Shuffle the DataFrame
    balanced_sample = balanced_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_sample



if __name__ == "__main__":
    file_path = 'C:\\Users\\tlang\\GitHub\\Introduction_to_DS\\datasets\\adult.csv'
    # loading the dataset from a CSV file
    df = pd.read_csv(file_path)
    
    df = process_data_discertization(df)

    n_samples = 2000
    df = balance_sample(df, n_samples)
    X = df.drop('income', axis=1)
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)

    # Merge X_train and y_train into one DataFrame
    df_train = pd.concat([X_train, y_train], axis=1)  

    # create Naive Bayes model
    nb = NaiveBayes('income')
    
    # call fit method with df_train
    nb.fit(df_train)

    predictions = nb.predict(X_test.to_numpy())
    # predictions = nb.predict(X_train.to_numpy())
    count_of_0 = predictions.count(0)
    count_of_1 = predictions.count(1)
    print("count_of_0: ", count_of_0)
    print("count_of_1: ", count_of_1)
    # evaluate the model
    evaluations = nb.evaluate(predictions, y_test)

    # Write evaluation results to a text file
    output_file_path = 'Naive_Bayes_results.txt'
    with open(output_file_path, 'a') as f:
        f.write(f"Regular prediction\n")
        f.write(f"train size: {df_train.shape[0]}\n")
        f.write(f"test size: {X_test.shape[0]}\n")
        for metric_name, metric_value in zip(["Accuracy", "Precision", "Recall", "F1 Score"], evaluations):
            f.write(f"{metric_name}: {metric_value:.4f}\n")
            
        f.write("\n")

    print(f"Results written to {output_file_path}")

    # Create a confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # Visualize the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 20})

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    