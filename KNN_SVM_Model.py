from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Load the Iris dataset
data = datasets.load_iris()
df = pd.DataFrame(data=np.c_[d['data'], d['target']], columns=d['feature_names'] + ['target'])
# Split the data
features = df.iloc[:, :-1]
target = df.iloc[:, -1]




# Standardize the features
scaler = StandardScaler().fit(features)
standardized_features = scaler.transform(features)

# Split the dataset into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(
    standardized_features, target, test_size=0.2, random_state=37)


# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(feature_train, target_train)

# Make predictions on the test set
target_pred = knn.predict(feature_test)


# Evaluate the model
accuracy = accuracy_score(target_test, target_pred)
precision = precision_score(target_test, target_pred, average='micro')
recall = recall_score(target_test, target_pred, average='micro')
f1 = f1_score(target_test, target_pred, average='micro')

# Print the evaluation metrics
print("KNN Model Results:")
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1 Score: {f1:.2f}")

svm = SVC(kernel='linear', random_state=37)
svm.fit(feature_train, target_train)

# Make predictions with the SVM model
svm_pred = svm.predict(feature_test)



# Evaluate the SVM model
svm_accuracy = accuracy_score(target_test, svm_pred)
svm_precision = precision_score(target_test, svm_pred, average='micro')
svm_recall = recall_score(target_test, svm_pred, average='micro')
svm_f1 = f1_score(target_test, svm_pred, average='micro')

# Print SVM model evaluation metrics
print("SVM Model Results")
print(f"Accuracy: {svm_accuracy:.2f}")
print(f"Precision: {svm_precision:.2f}")
print(f"Recall: {svm_recall:.2f}")
print(f"F1 Score: {svm_f1:.2f}")
