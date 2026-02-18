import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from io import StringIO

# ----------------------------
# 1️⃣ CSV Data Embedded
# ----------------------------
csv_data = """
Roll_No,Study_Hours,Attendance,Sleep_Hours,Previous_Marks,Final_Marks
1,1,50,8,40,42
2,2,60,7,45,48
3,3,65,6,50,52
4,4,70,6,55,58
5,5,75,7,60,63
6,6,80,5,65,68
7,7,85,6,70,74
8,8,90,7,75,78
9,9,95,8,80,85
10,10,100,5,85,90
"""

# Read CSV from string
df = pd.read_csv(StringIO(csv_data))

# ----------------------------
# 2️⃣ Define Features & Target
# ----------------------------
X = df[['Study_Hours','Attendance','Sleep_Hours','Previous_Marks']]
y = df['Final_Marks']

# ----------------------------
# 3️⃣ Split Data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 4️⃣ Train Model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# 5️⃣ Model Accuracy
# ----------------------------
print("Model Accuracy (R2 Score):", r2_score(y_test, model.predict(X_test)))

# ----------------------------
# 6️⃣ User Input via Roll Number
# ----------------------------
roll = int(input("Enter Student Roll Number: "))

# Check if Roll No exists
if roll in df['Roll_No'].values:
    student = df[df['Roll_No'] == roll]
    
    # Display student details
    print("\nStudent Details:")
    print(student[['Roll_No','Study_Hours','Attendance','Sleep_Hours','Previous_Marks']])
    
    # Prepare input for prediction
    student_features = student[['Study_Hours','Attendance','Sleep_Hours','Previous_Marks']]
    predicted_marks = model.predict(student_features)
    
    print("Predicted Final Marks:", round(predicted_marks[0], 2))
else:
    print("Roll Number not found in dataset!")


