import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Step 1: Create Dataset
data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Step 2: Encode Data
df_encoded = df.copy()
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df_encoded.drop('BuyCoffee', axis=1)
y = df_encoded['BuyCoffee']

# Step 3: Train Model
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Step 4: Streamlit UI
st.title("☕ Buy Coffee Predictor")
st.sidebar.header("Enter Your Conditions")

def user_input():
    weather = st.sidebar.selectbox("Weather", df['Weather'].unique())
    time = st.sidebar.selectbox("Time of Day", df['TimeOfDay'].unique())
    sleep = st.sidebar.selectbox("Sleep Quality", df['SleepQuality'].unique())
    mood = st.sidebar.selectbox("Mood", df['Mood'].unique())
    return pd.DataFrame([[weather, time, sleep, mood]], columns=['Weather', 'TimeOfDay', 'SleepQuality', 'Mood'])

input_df = user_input()

# Encode user input
input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoders[col].transform(input_encoded[col])

# Predict
prediction = model.predict(input_encoded)[0]
prediction_label = label_encoders['BuyCoffee'].inverse_transform([prediction])[0]

st.subheader("🧠 Prediction Result:")
st.success(f"The model predicts: **{prediction_label}**")

st.subheader("📥 Your Input:")
st.write(input_df)

# Optional: Plot the decision tree
st.subheader("🌳 Decision Tree")
fig, ax = plt.subplots(figsize=(10, 5))
plot_tree(model, feature_names=X.columns, class_names=label_encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)

# Explain path
st.subheader("🧭 Explanation")
st.markdown("""
The decision tree above shows how the model splits based on features like:
- **Mood** and **SleepQuality** are most important.
- If you are *Tired* with *Poor* sleep, you're more likely to buy coffee.
""")
