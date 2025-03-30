import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


def run():
    
    heart_df = pd.read_csv('Datasets/heart.csv')
    
    st.title('Heart Checkup')
    st.sidebar.header('Patient Data')
    st.subheader('Training Data Stats')
    st.write(heart_df.describe())

    x = heart_df.drop(['target'], axis=1)
    y = heart_df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    def user_report():
        age = st.sidebar.slider('Age', 29, 77, 50)
        sex = st.sidebar.radio('Sex', [0, 1])
        cp = st.sidebar.slider('Chest Pain Type', 0, 3, 1)
        trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 120)
        chol = st.sidebar.slider('Cholesterol', 126, 564, 240)
        fbs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', [0, 1])
        restecg = st.sidebar.slider('Resting Electrocardiographic Results', 0, 2, 1)
        thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
        exang = st.sidebar.radio('Exercise Induced Angina', [0, 1])
        oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 3.0)
        slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment', 0, 2, 1)
        ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
        thal = st.sidebar.slider('Thal', 0, 3, 2)

        user_report_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        report_data = pd.DataFrame(user_report_data, index=[0])
        return report_data

    user_data = user_report()
    st.subheader('Patient Data')
    st.write(user_data)

    rf  = RandomForestClassifier()
    rf.fit(x_train, y_train)
    user_data.columns = user_data.columns.str.upper()
    user_data_np = user_data.to_numpy()
    user_result = rf.predict(user_data_np)    
    
    st.subheader('Your Report:')
    output = 'No Heart Disease' if user_result[0] == 0 else 'Heart Disease'
    st.title(output)
    st.subheader('Accuracy:')
    st.write(str(accuracy_score(y_test, rf.predict(x_test)) * 100) + '%')

    # Visualization Section
    st.subheader('Visualizations')
    
    # Select feature to compare with age
    feature = st.selectbox('Select a feature to compare with age', 
                           ['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                            'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    st.write(f'Plotting {feature} vs Age')

     # Plot the training data
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=heart_df['age'], y=heart_df[feature], hue=heart_df['target'].map({0: 'No Heart Disease', 1: 'Heart Disease'}), palette='coolwarm')
    
    # Highlight user data
    user_feature_value = user_data[feature.upper()].values[0]
    plt.scatter(user_data['AGE'].values[0], user_feature_value, color='black', s=100, label='Your Input', edgecolor='white')

    plt.xlabel('Age')
    plt.ylabel(feature.capitalize())
    plt.title(f'{feature.capitalize()} vs Age')
    plt.legend()
    st.pyplot(plt)

run()