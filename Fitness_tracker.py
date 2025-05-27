import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')

st.toast("Keep working hard")

st.write("# Personal Fitness Tracker")
with st.expander("About This WebApp"):
    st.write("This Fitness tarcker is just a cheap model for those who can't afford much money on fitness bands and who were so enthusiastic to check out how much calories they've burned and many .")
    st.header("Key Features :")
    st.write("1. User input form ")
    st.write("2. Machine Learning model (RandomForestRegressor)")
    st.write("3. Plotly Scatter plot")
    st.write("4. Fitness Insights")
    st.write("5. Model Performance evaluations")
with st.expander("How to use the WebApp"):
    st.write("1. In the left side of page ,you will see a sidebar go for it")
    st.write("2. In that Enter your Details")
    st.write("3. Adjust your details according to you")
    st.write("4. Go For It")
st.sidebar.header("Enter your data")
if "age" not in st.session_state:
    st.session_state.age = 25
if "gender" not in st.session_state:
    st.session_state.gender = "Male"
if "bmi" not in st.session_state:
    st.session_state.bmi = 25.0

def user_input_features():
    st.session_state.age = st.sidebar.number_input("Enter Age", min_value=10, max_value=100, value=st.session_state.age)
    st.session_state.gender = st.sidebar.selectbox("Select Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
    st.session_state.bmi = st.sidebar.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=st.session_state.bmi)
    duration = st.sidebar.slider("Daily WorkOut Duration (min): ", 0, 30, 15)
    heart_rate = st.sidebar.slider("Heart Rate during Workout: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature(C) during WorkOut : ", 36, 42, 41)
    gender_encoded = 1 if st.session_state.gender == "Male" else 0

    data_model = {
        "Age": st.session_state.age,
        "BMI": st.session_state.bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender": gender_encoded
    }

    features = pd.DataFrame([data_model])
    return features
df = user_input_features()
    
st.write("---")
with st.expander("Your Parameters"):
    st.write(df)
st.write("---")

calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories,on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)


exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories",axis=1)
y_test = exercise_test_data["Calories"]

random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

st.write("---")
st.write("### Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories Burned Today as per your data**")


st.write("---")
st.write("### Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                           (exercise_df["Calories"] <= calorie_range[1])]
if not similar_data.empty:
    st.write("###  People with Similar Calorie Burn:")
    st.write(similar_data.sample(min(5, len(similar_data))))
else:
    st.write("⚠️ No similar results found in the dataset.")

st.write("---")
st.header("General Information: ")

boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")

st.write("---")
y_pred = random_reg.predict(X_test)

st.write("### R² Score of the Model")
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)
st.write(f"R² Score: {round(r2, 2)}")

if r2 >= 0.9:
    st.write("Excellent Model: This Model is very Good Model so that everyone will use it efficiently.")
elif r2 >= 0.75:
    st.write("Good Model: Explains a significant portion of the variance.")
elif r2 >= 0.5:
    st.write("Average Model: Still usable but could be improved.")
elif r2 >= 0:
    st.write("Poor Model: Explains very little variance.")
else:
    st.write("Very Bad Model: Worse than a random guess!")

st.write("## Thank You 💖")
