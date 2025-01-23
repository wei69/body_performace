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
        background-image: linear-gradient(to bottom, #6dd5ed, #2193b0, #1e3c72, #3f51b5);
        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        font-family: 'Arial', sans-serif;
    }
    .stTabs [role="tablist"] {
        display: flex;
        border-bottom: 2px solid #ddd;
        margin-bottom: 1em;
    }
    .title {
        color: white;
    }
    .stTabs [role="tab"] {
        flex: 1;
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #ddd;
        border-bottom: none;
        cursor: pointer;
        background: #f8f9fa;
        color: #333;
        border-radius:12px;
        transition: background 0.3s, color 0.3s;
         
    }
    .stTabs [role="tab"][aria-selected="true"] {
    background: #007bff;  /* Highlighted tab color */
    color: white;          /* Text color for selected tab */
}
    .stTabs [role="tab"]:hover {
        background: red;
        color:white;
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

# Set title with custom HTML
st.markdown(
    """
    <h1 style="color: white; text-align: center;">Predicting Your Body Performance Metrics</h1>
    """, unsafe_allow_html=True
)

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

        # Display the fitness prediction with white color
        st.markdown(
            """
            <h3 style="color: white;">Your Body-performance Prediction</h3>
            """, unsafe_allow_html=True
        )

        # Modify prediction label to match fitness categories (A = fittest, D = least fit)
        performance_categories = {
            'A': '💪 Fittest',
            'B': '👍 Above Average',
            'C': '🧑‍🦱 Average',
            'D': '⚡ Least Fit'
        }

        st.markdown(
            f"""
            <h3 style="color: white;">Fitness Category: {performance_categories.get(prediction[0], 'Unknown')}</h3>
            """, unsafe_allow_html=True
        )

        # Visualization options
                # Visualization options
        tab1, tab2, tab3 = st.tabs(["Fitness metrics📊", "Body fats📈", "Prediction Probabilities🔢"])

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

        with tab3:
            # Ensure the model supports prediction probabilities
            try:
                prediction_probs = model.predict_proba(scaled_input)[0]
                categories = ['A (Fittest)', 'B (Above Average)', 'C (Average)', 'D (Least Fit)']

                prob_data = pd.DataFrame({
                    'Category': categories,
                    'Probability (%)': prediction_probs * 100
                })

                prob_chart = alt.Chart(prob_data).mark_bar().encode(
                    x='Category:O',
                    y='Probability (%):Q',
                    color='Category:N'
                ).properties(
                    title="Prediction Probabilities"
                )
                st.altair_chart(prob_chart, use_container_width=True)
            except AttributeError:
                st.error("The model does not support probability predictions.")

