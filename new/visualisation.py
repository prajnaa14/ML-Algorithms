import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\online_course_engagement_data.csv")

# Step 2: Pairplot to visualize relationships and distributions
sns.pairplot(data, vars=['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores'], hue='CourseCompletion')
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

# Step 3: Histograms to visualize distributions
data.hist(figsize=(10, 8), bins=20, grid=False)
plt.suptitle('Histograms of Variables', y=1.02)
plt.tight_layout()
plt.show()

# Step 4: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data[['TimeSpentOnCourse', 'NumberOfVideosWatched', 'NumberOfQuizzesTaken', 'QuizScores', 'CourseCompletion']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
