import pandas as pd # For data manipulation and analysis
from sklearn.model_selection import train_test_split    # For splitting the dataset
from sklearn.tree import DecisionTreeClassifier # For building the model
import joblib   # For saving the trained model

# Load the dataset
gym_data = pd.read_csv('gym_stats.csv')

# Prepare the dataset
x = gym_data.drop('Workout_Type', axis=1)   # Separate target column to get features
y = gym_data['Workout_Type']

# Convert categorical variables to numeric using one-hot encoding
x = pd.get_dummies(x, drop_first=True)

# Split the dataset, 80% for training and 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model using DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'decision_tree_model.pkl')