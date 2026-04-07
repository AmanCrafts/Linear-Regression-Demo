# Create a synthetic dataset for linear regression with 5 columns and 1000 rows
import pandas as pd
import numpy as np

np.random.seed(42)

rows = 1000

data = pd.DataFrame({
    "Hours_Studied": np.random.uniform(0, 15, rows).round(2),
    "Sleep_Hours": np.random.uniform(4, 9, rows).round(2),
    "Attendance_Percentage": np.random.uniform(50, 100, rows).round(2),
    "Previous_Score": np.random.uniform(40, 95, rows).round(2),
})

# Create target with some linear relation + noise
data["Exam_Score"] = (
    2.5 * data["Hours_Studied"] +
    1.2 * data["Sleep_Hours"] +
    0.3 * data["Attendance_Percentage"] +
    0.5 * data["Previous_Score"] +
    np.random.normal(0, 5, rows)
).round(2)

# Save to CSV
file_path = "data/linear_regression_mock_dataset.csv"
data.to_csv(file_path, index=False)

file_path