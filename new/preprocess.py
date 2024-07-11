import pandas as pd

# Load the data
df = pd.read_csv("C:\\Users\\PRAJNA\\Downloads\\online_course_engagement_data.csv")

# Display the first few rows of the dataframe
print(df.head())

# Check the dimensions of the dataframe
print(df.shape)

# Check the data types of each column
print(df.dtypes)

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()