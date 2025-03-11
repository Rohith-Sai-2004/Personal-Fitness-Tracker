import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')

st.write("# Personal Fitness Tracker")
st.write("**It is a personal fitness tracker used to track our fitness via workout comparisons among people. I have already inserted data from several individuals, so you can easily check where your standards are.**")
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
    body_temp = st.sidebar.slider("Body Temperature(C) during WorkOut : ", 36, 42, 38)
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
st.write("### Your Parameters :")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
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

st.write("### Workout Duration vs. Calories Burned")
st.write("**This scatter plot had taken the data from the data sets and from you.**")
st.write("**This plot had the attributes calories burned and duration of workout .We see that there is a point in red color that shows where we are among others.**")
fig, ax = plt.subplots(facecolor='black')  
ax.set_facecolor('#1e1e1e')
ax.scatter(exercise_df["Duration"], exercise_df["Calories"], alpha=0.5, color='cyan', label="Other Users")
ax.scatter(df["Duration"].values[0], prediction[0], color='red', marker='o', s=100, label="Your Data")  
ax.set_xlabel("Workout Duration (min)", color='white')
ax.set_ylabel("Calories Burned", color='white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
legend = ax.legend()
for text in legend.get_texts():
    text.set_color("white")
st.pyplot(fig)



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

# Interpretation
if r2 >= 0.9:
    st.write("🚀 Excellent Model: This Model is very Good Model so that everyone will use it efficiently.")
elif r2 >= 0.75:
    st.write("👍 Good Model: Explains a significant portion of the variance.")
elif r2 >= 0.5:
    st.write("⚠️ Average Model: Still usable but could be improved.")
elif r2 >= 0:
    st.write("❌ Poor Model: Explains very little variance.")
else:
    st.write("❌ Very Bad Model: Worse than a random guess!")
