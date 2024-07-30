import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the app
st.title('Customer Churn Prediction')
st.markdown("""
    **Welcome to the Customer Churn Prediction App!**

    This application predicts whether a customer will churn (leave) based on various features such as their country, gender, age, balance, and more.

    **How to Use:**
    - Use the sidebar to input the customer's details.
    - Your prediction will automatically load.
    
    The model was trained on historical data to provide insights into customer behavior and help businesses improve their retention strategies.
""")
# Sidebar header
st.sidebar.header('User Input Features')

# Function to preprocess the data
def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna()

    # Initialize LabelEncoders for categorical columns
    label_encoders = {}
    categorical_columns = ['country', 'gender', 'credit_card', 'active_member']

    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le
        df[col] = le.transform(df[col])

    return df, label_encoders

# Load the data
data = pd.read_csv(r'data.csv')


# Check if the necessary columns exist in the dataset
required_columns = ['country', 'gender', 'credit_card', 'active_member', 'credit_score', 'customer_id', 'products_number']
for col in required_columns:
    if col not in data.columns:
        st.error(f"Column '{col}' does not exist in the dataset.")
        st.stop()

# Preprocess the data
data, label_encoders = preprocess_data(data)

# Define features and target variable
target_column = 'churn'
if target_column not in data.columns:
    st.error(f"Target column '{target_column}' does not exist in the dataset.")
    st.stop()

X = data.drop(columns=[target_column])
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy in the Streamlit app
st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy:.2f}')

# Function to get user input features
def user_input_features():
    country = st.sidebar.selectbox('Country', ('France', 'Spain', 'Germany'))
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 18, 100, 35)
    tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    balance = st.sidebar.number_input('Balance', 0.0, 250000.0, 50000.0)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 2)
    credit_card = st.sidebar.selectbox('Has Credit Card', ('Yes', 'No'))
    active_member = st.sidebar.selectbox('Active Member', ('Yes', 'No'))
    estimated_salary = st.sidebar.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)
    
    data = {
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'num_of_products': num_of_products,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }
    features = pd.DataFrame(data, index=[0])
    
    # Encode the categorical features using the same encoders as the training data
    for col in ['country', 'gender', 'credit_card', 'active_member']:
        if col in label_encoders:
            encoder = label_encoders[col]
            if col in ['credit_card', 'active_member']:
                features[col] = features[col].map({'Yes': 1, 'No': 0})
            else:
                features[col] = encoder.transform(features[col])
        else:
            st.error(f"Label encoder for column {col} not found.")
            return pd.DataFrame()  # Return empty DataFrame

    # Ensure input_df has all the required columns
    for col in X.columns:
        if col not in features.columns:
            features[col] = 0  # Fill missing columns with default value (0)

    # Reorder columns to match the training data
    features = features[X.columns]

    return features

# Get user input
input_df = user_input_features()

if not input_df.empty:
    # Scale the user input
    input_df = scaler.transform(input_df)
    
    # Predict the churn
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display the prediction and prediction probability
    st.subheader('Prediction')
    churn = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Churn Prediction: {churn}')

    st.subheader('Prediction Probability')
    st.write(f'Churn Probability: {prediction_proba[0][1]:.2f}')
