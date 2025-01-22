import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# Add background image via custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://shotkit.com/wp-content/uploads/2021/08/Fitness-photography-anush-gorak.jpeg');  /* You can also use a local path */
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        color: white; 
    }
    .stTabs .stTab {
      color: white;
    }

    .stTabs .stTab:hover {
        background-color: lightgray;
    }
    </style>
    """, unsafe_allow_html=True
)

# Function to render custom metric
def custom_metric(label, value, delta=None):
    delta_html = f"<span style='color:white;'>{delta}</span>" if delta else ""
    st.markdown(
        f"""
        <div style="color: white; font-size: 1.5em; margin-bottom: 10px;">
            <strong>{label}</strong>
        </div>
        <div style="color: white; font-size: 2.5em;">
            {value} {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )

# Load the pre-trained model and scaler
model = joblib.load('gl.pkl')
scaler = joblib.load('scaler.pkl')

# Function to capture user inputs
def get_user_input():
    age = st.sidebar.slider('Age', 20, 64, 35)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    height = st.sidebar.slider('Height (cm)', 140, 200, 170)
    weight = st.sidebar.slider('Weight (kg)', 40, 120, 70)
    body_fat_percent = st.sidebar.slider('Body Fat Percentage (%)', 5, 50, 20)
    grip_force = st.sidebar.slider('Grip Force (kg)', 1, 100, 50)
    sit_and_bend_forward = st.sidebar.slider('Sit and Bend Forward (cm)', 0, 100, 20)
    sit_ups = st.sidebar.slider('Sit-ups Count', 0, 100, 30)
    broad_jump = st.sidebar.slider('Broad Jump (cm)', 0, 300, 200)

    # Derived features
    body_fat_category_high = body_fat_percent > 27
    body_fat_category_low = body_fat_percent < 17
    body_fat_category_normal = not (body_fat_category_high or body_fat_category_low)

    gender_numerical = 0 if gender == 'Female' else 1

    # Create DataFrame for input data
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender_numerical],
        'height': [height],
        'weight': [weight],
        'body fat_%': [body_fat_percent],
        'gripForce': [grip_force],
        'sit and bend forward_cm': [sit_and_bend_forward],
        'sit-ups counts': [sit_ups],
        'broad jump_cm': [broad_jump],
        'body_fat_category_High': [int(body_fat_category_high)],
        'body_fat_category_Low': [int(body_fat_category_low)],
        'body_fat_category_Normal': [int(body_fat_category_normal)]
    })

    return input_data

# Get user input
user_input = get_user_input()
st.title("Predicting Your Body Performance Metrics")
# Check for missing values
if np.any(user_input.isnull().values):
    st.error("Input data contains missing values. Please check your inputs.")
else:
    # Scale the input data
    input_array = user_input.to_numpy()
    scaled_input = scaler.transform(input_array)

    # Make predictions
    try:
        prediction = model.predict(scaled_input)
    except Exception as e:
        st.error(f"Error in making prediction: {str(e)}")
        prediction = None

    if prediction:
        # Display the results in columns using custom HTML metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            custom_metric("🏋️ Weight", f"{user_input['weight'].iloc[0]} kg")

        with col2:
            custom_metric("📏 Height", f"{user_input['height'].iloc[0]} cm")

        with col3:
            body_fat_percent = user_input['body fat_%'].iloc[0]
            if body_fat_percent > 27:
                custom_metric("💪 Body Fat Percentage", f"{body_fat_percent}%", "🔥 High")
            elif body_fat_percent < 17:
                custom_metric("💪 Body Fat Percentage", f"{body_fat_percent}%", "❄️ Low")
            else:
                custom_metric("💪 Body Fat Percentage", f"{body_fat_percent}%", "✅ Normal")

        # Title and prediction
   
        st.subheader("**Your Body-performance Prediction**")
        
        # Modify prediction label to match fitness categories (A = fittest, D = least fit)
        performance_categories = {
            'A': '💪 Fittest',
            'B': '👍 Above Average',
            'C': '🧑‍🦱 Average',
            'D': '⚡ Least Fit'
        }
        
        st.subheader(f"**Fitness Category: {performance_categories.get(prediction[0], 'Unknown')}**")

        # Visualization options
        tab1, tab2 = st.tabs(["Bar Chart📊", "Pie Chart 📈"] )

        with tab1:
            metrics = ['Grip Force', 'Sit-ups Count', 'Broad Jump', 'Body Fat %']
            values = [
                user_input['gripForce'].iloc[0],
                user_input['sit-ups counts'].iloc[0],
                user_input['broad jump_cm'].iloc[0],
                user_input['body fat_%'].iloc[0]
            ]
            
            bar_data = pd.DataFrame({'Metric': metrics, 'Value': values})
            bar_chart = alt.Chart(bar_data).mark_bar().encode(
                x='Metric:O',
                y='Value:Q',
                color='Metric:N'
            ).properties(
                title="Comparison of Fitness Metrics"
            )
            st.altair_chart(bar_chart, use_container_width=True)

        with tab2:
            pie_data = pd.DataFrame({
                'Category': ['High Body Fat', 'Normal Body Fat', 'Low Body Fat'],
                'Value': [
                    user_input['body_fat_category_High'].iloc[0],
                    user_input['body_fat_category_Normal'].iloc[0],
                    user_input['body_fat_category_Low'].iloc[0]
                ]
            })

            if pie_data['Value'].sum() == 0:
                st.write("No data available for body fat categories.")
            else:
                pie_chart = alt.Chart(pie_data).mark_arc().encode(
                    theta='Value',
                    color='Category'
                ).properties(
                    title="Body Fat Categories"
                )
                st.altair_chart(pie_chart, use_container_width=True)
