#Decision tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\\Users\\PRAJNA\\Downloads\\online_course_engagement_data.csv"
data = pd.read_csv(file_path)

# Define the target variable and features
target = 'CourseCompletion'
features = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores', 'DeviceType', 'CourseCategory']

# Define the preprocessing steps for numerical and categorical features
numerical_features = ['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores']
categorical_features = ['DeviceType', 'CourseCategory']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree model within a pipeline with max_depth=3
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=2))])

model.fit(X_train, y_train)

# Plot the Decision Tree with maximum depth of 3
plt.figure(figsize=(15, 10))
plot_tree(model.named_steps['classifier'], filled=True, class_names=['Not Completed', 'Completed'], feature_names=numerical_features + categorical_features)
plt.title("Decision Tree Visualization (Max Depth = 2)")
plt.show()

# Print the text representation of the tree
text_representation = export_text(model.named_steps['classifier'], feature_names=numerical_features + categorical_features)
print(text_representation)


