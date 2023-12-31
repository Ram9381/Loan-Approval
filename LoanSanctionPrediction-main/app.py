from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import jsonify
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

# Load your loan prediction model and data here
interest_rates = pd.read_csv("interest_rates.csv")
interest_rates.dropna(inplace=True)
# Assume 'Loan_Status' is the target variable
data = pd.read_csv("LoanApprovalPrediction.csv")
data = data.drop(['Loan_ID','Gender','Married','Property_Area'], axis=1)
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
data.dropna(inplace=True)
X=data.drop('Loan_Status',axis=1)
y = data['Loan_Status']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        applicant_income = float(request.form.getlist('applicant_income')[0])
        coapplicant_income = float(request.form.getlist('coapplicant_income')[0])
        loan_amount = float(request.form.getlist('loan_amount')[0])
        loan_term = float(request.form.getlist('loan_term')[0])
        credit_history = float(request.form.getlist('credit_history')[0])
        loan_type = request.form.getlist('loan_type')[0]
        education = int(request.form['Education'])
        Self_Employed  = int(request.form.getlist('Self_Employed')[0])
        Dependents = float(request.form.getlist('Dependents')[0])
        # Retrieve the interest rate based on the selected loan type
        interest_rate = interest_rates.loc[interest_rates['loan_type'] == loan_type, 'interest_rate'].values
        interest_rate = interest_rate[0] if len(interest_rate) > 0 else 0  # Default to 0 if loan type not found

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Dependents': [Dependents],
            'Education' : [education],
            'Self_Employed' : [Self_Employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            #'Interest_Rate': [interest_rate]
        })
        print(input_data)
        # Make a prediction using the model
        prediction = model.predict(input_data)
        print(prediction,prediction[0])
        prediction_result = 'Approved' if prediction[0] == 1 else 'Not Approved'
        return render_template('result.html', prediction=prediction_result, interest_rate=interest_rate, loan_types=interest_rates['loan_type'])

if __name__ == '__main__':
    app.run(debug=True)