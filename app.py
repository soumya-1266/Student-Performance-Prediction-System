from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('best_student_model.pkl')


def predict_result(student):
    return model.predict(pd.DataFrame([student]))[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    student = {
        'Study_Hours': float(request.form['study_hours']),
        'Attendance': float(request.form['attendance']),
        'Previous_Marks': float(request.form['previous_marks']),
        'Assignments': float(request.form['assignments']),
        'Internal_Marks': float(request.form['internal_marks']),
        'Income_Group': request.form['income_group'],
        'Parent_Education': request.form['parent_education']
    }

    result = predict_result(student)

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(debug=True)