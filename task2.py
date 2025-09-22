# Import necessary libraries
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Encode labels for compatibility (optional with sklearn recent versions)
iris['species'] = iris['species'].astype('category').cat.codes

# Define features and target
X = iris.drop('species', axis=1)
y = iris['species']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
y_pred_dtree = dtree.predict(X_test)

# Evaluate results
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dtree))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dtree))

# Confusion matrix visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_log_reg), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
sns.heatmap(confusion_matrix(y_test, y_pred_dtree), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Decision Tree Confusion Matrix')
plt.show()

# Visualize pairplot for data
sns.pairplot(data=iris, hue='species')
plt.show()
