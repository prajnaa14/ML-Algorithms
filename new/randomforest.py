import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
file_path = "C:\\Users\\PRAJNA\\Downloads\\online_course_engagement_data.csv"
data = pd.read_csv(file_path)

# Step 2: Prepare the data
X = data[['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores']]  # Features
y = data['CourseCompletion']  # Target variable (binary classification)

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest Model with Maximum Depth of 2
model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")
print(classification_report(y_test, y_pred))

# Step 6: Display Random Forest Properties
print("Random Forest properties:")
print(f"Number of trees in the forest: {len(model.estimators_)}")
print(f"Feature importance:\n{model.feature_importances_}")
