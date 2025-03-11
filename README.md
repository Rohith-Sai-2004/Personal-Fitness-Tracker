# Personal Fitness Tracker

This **Streamlit-based web application** predicts the number of **kilocalories burned** based on user inputs such as **Age, BMI, Duration, Heart Rate, Body Temperature, and Gender**. It also provides statistical comparisons and similar records from an existing dataset.

## How to Run the fitness WebApp
Just click on the link :   
https://personal-fitness-tracker-run.streamlit.app/

## Features
✅ User-friendly **sidebar input** for entering fitness parameters.  
✅ **Calorie prediction** using a trained **RandomForestRegressor** model.  
✅ **Statistical comparison** with other users' data.  
✅ **Similar results** based on calorie ranges.

## How It Works
1. Enter your fitness details in the sidebar.
2. The app predicts the number of **kilocalories burned**.
3. Compare your stats with other users.
4. See similar fitness records for reference.
5. We considered input for Age ,BMI ,Gender as somewhat constant so that we used session_state and for heart_rate,body_temp,Duration is slider because it will vary day by day.

## Tech Used
🔹 **Python** (Pandas, Scikit-learn)  
🔹 **Streamlit** (for web-based UI)  
🔹 **Random Forest Regression** (for prediction)  
🔹 **Matplotlib** (for future data visualization)  


