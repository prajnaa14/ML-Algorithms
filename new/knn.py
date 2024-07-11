import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
data = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\online_course_engagement_data.csv")

# Step 2: Prepare the data
X = data[['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores']]  # Features
y = data['CourseCompletion']  # Target variable (binary classification)

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the KNN Model and Evaluate
model = KNeighborsClassifier(n_neighbors=5)  # Initialize KNN with 5 neighbors (adjust as needed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Step 5: Calculate accuracy and display classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")
print(classification_report(y_test, y_pred))
