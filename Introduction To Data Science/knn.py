from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Euclidean Distance with numpy
def euclidean_distance_np(p1,p2):
    dist = np.linalg.norm(p1-p2)
    return dist

#Euclidean Distance
def euclidean_distance(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist

# Calculating the manhattan distance between two vectors
def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(np.abs(x1 - x2))

# Calculating the minkowski distance between two vectors
def minkowski_distance(x1: np.ndarray, x2: np.ndarray, p: int) -> float:
    return np.sum(np.abs(x1 - x2)**p)**(1/p)

# Calculating the cosine similarity between two vectors
def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# Calculating the hamming distance between two vectors
def hamming_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sum(x1 != x2)


# def get_data_with_pca(path: str, num_of_samples: int, random_state: int, n_components=2):
def get_data_with_pca(random_samples: pd.DataFrame, n_components=2):    
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    random_samples['income'] = label_encoder.fit_transform(random_samples['income'])

    # Splitting the dataset into features (X) and target variable (y)
    X = random_samples.drop('income', axis=1)  # Features
    y = random_samples['income'].to_numpy()  # Target variable

    # Define categorical_columns here (you missed this part in your code)
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']

    # Perform one-hot encoding
    df_one_hot_encoded = pd.get_dummies(X, columns=categorical_columns)

    # Identify columns with more than two unique values (non-binary)
    non_binary_columns = [col for col in df_one_hot_encoded.columns if df_one_hot_encoded[col].nunique() > 2]

    # Normalize only non-binary columns using StandardScaler
    scaler = StandardScaler()
    df_normalized = df_one_hot_encoded.copy()
    df_normalized[non_binary_columns] = scaler.fit_transform(df_one_hot_encoded[non_binary_columns])

    # Perform PCA
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df_normalized)
    print("df_pca.shape[0]: ", df_pca.shape[0])
    print("df_pca.shape[1]: ", df_pca.shape[1])
    # Convert the DataFrame to a NumPy array
    y = random_samples['income'].to_numpy()

    return df_pca, y

    
class knn_classifier:
    def __init__(self, k: int, distance_metric):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.num_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray, c: int):
        self.X_train = X
        self.y_train = y
        self.num_classes = c

    def predict(self, vlues_to_predict: np.ndarray) -> np.ndarray:
        predicted_labels = []
        
        for y in vlues_to_predict:
            
            dist_arr = []
            
            for i in range(len(self.X_train)):
                dist = self.distance_metric(np.array(self.X_train[i,:]), y)
                dist_arr.append((dist, i))
                if len(dist_arr) > self.k:
                    
                    # Sort the list and remove the element with the biggest distance
                    dist_arr.sort(key=lambda x: x[0], reverse=False)
                    dist_arr.pop()
            # Count the number of each class in the k closest points
            # Create a dictionary to store the count of each class
            class_count = {}
            for i in range(self.num_classes):
                class_count[i] = 0
                    # Initialize the count dictionary
            for i in range(len(dist_arr)):
                index = dist_arr[i][1]
                class_count[self.y_train[index]] += 1      

            predicted_labels.append(max(class_count, key=class_count.get))      
  
      
        return np.array(predicted_labels)
    
    # Receives the predictions and the actual values and returns tuple with the 
    # accuracy, precision, recall, and f-measure
    def evaluate(self, predictions, y_test):
        evaluations = []
        acc = accuracy_score(y_test, predictions)
        print("accuarcy: ", acc)
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

        return acc, ps, rs, f1
    
def balance_sample(df: pd.DataFrame, n_samples: int):
    
    # Separate the data into two classes
    class_0 = df[df['income'] == '<=50K']
    class_1 = df[df['income'] == '>50K']

    # # Sample half the size from each class
    # n_samples = min(len(class_0), len(class_1))
 

    # Use sample function to get random samples
    sampled_class_0 = class_0.sample(n=n_samples, random_state=42)
    sampled_class_1 = class_1.sample(n=n_samples, random_state=42)

    # Concatenate the sampled dataframes
    balanced_sample = pd.concat([sampled_class_0, sampled_class_1])

    # Shuffle the DataFrame
    balanced_sample = balanced_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_sample

def cluster_by_class(X, y, n_clusters_per_class):
    unique_classes = np.unique(y)
    clustered_features = []
    clustered_labels = []

    for class_label in unique_classes:
        # Filter data for the current class
        class_data = X[y == class_label]

        # Apply K-Means clustering for the current class
        kmeans = KMeans(n_clusters=n_clusters_per_class)
        cluster_assignments = kmeans.fit_predict(class_data)

        # Represent each cluster by its centroid
        for i in range(n_clusters_per_class):
            cluster_samples = class_data[cluster_assignments == i]
            cluster_centroid = np.mean(cluster_samples, axis=0)

            # Append clustered features and corresponding class label
            clustered_features.append(cluster_centroid)
            clustered_labels.append(class_label)

    return np.array(clustered_features), np.array(clustered_labels)


if __name__ == "__main__":
    path_adults = 'C:\\Users\\tlang\\GitHub\\Introduction_to_DS\\datasets\\adult.csv'

    # importing the data with pandas
    df = pd.read_csv(path_adults)
    
    # Handling missing values marked as '?'
    df.replace('?', pd.NA, inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)
    samples_per_class = 1500
    # Extract balanced random samples
    random_samples = balance_sample(df, samples_per_class)
    
    n_components = 6
    X, y = get_data_with_pca(random_samples, n_components=n_components)
    # X, y = cluster_by_class(X, y, 1500)

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    k = 63
    knn = knn_classifier(k, euclidean_distance_np)    
    knn.fit(X_train, y_train, 2)
    pred = knn.predict(X_test)
    eva = knn.evaluate(y_test, pred)

    # # for loop for different values of k:
    # evaluation = []
    # K = []
    # k = 1
    # # for i in range(1, 52):
    # for i in range(1, 52):
    #     print("K: ", k)
    #     K.append(k)
    #     knn = knn_classifier(k, euclidean_distance_np)    
    #     knn.fit(X_train, y_train, 2)
    #     pred = knn.predict(X_test)
    #     eva = knn.evaluate(y_test, pred)
        
    #     evaluation.append(("K=" + str(k), eva))
    #     k += 2
        

    # # Write evaluation results to a text file
    # output_file_path = 'evaluation_results_cluster_balanced_sampels_3000.txt'
    # with open(output_file_path, 'w') as f:
    #     if n_components != 0:
    #         f.write(f"PCA components: {n_components}\n")
    #     f.write(f" num_of_samples: 3000 balanced clustering\n")
    #     # f.write(f" random_state: {random_state}\n")
    #     for k, metrics_tuple in zip(K, evaluation):
    #         f.write(f"K={k}\n")
    #         for metric_name, metric_value in zip(["Accuracy", "Precision", "Recall", "F1 Score"], metrics_tuple[1]):
    #             f.write(f"{metric_name}: {metric_value:.4f}\n")
    #         f.write("\n")

    # print(f"Results written to {output_file_path}")

    # # Extract accuracy values from the evaluation results
    # accuracies = [eva[0] for _, eva in evaluation]

    # # Get the min and max values
    # min_accuracy = min(accuracies)
    # max_accuracy = max(accuracies)
    # print("min_accuracy: ", min_accuracy)
    # print("max_accuracy: ", max_accuracy)
    # # Set a wider figure size
    # plt.figure(figsize=(12, 6))

    # # Plotting the graph
    # plt.plot(K, accuracies, linestyle='-', linewidth=2, color='blue')
    # plt.title('KNN Model Accuracy for Different Values of K')
    # plt.xlabel('K')
    # plt.ylabel('Accuracy')

    # # Customize the tick labels on the x-axis
    # plt.xticks(K, fontsize=8)

    # # Set y-axis limits to range from min_accuracy to 1.0
    # plt.ylim(min_accuracy, 1.0)

    # plt.grid(False)
    # plt.show()

  # Create a confusion matrix
    cm = confusion_matrix(y_test, pred)

    # Visualize the confusion matrix using a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 20})

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    
