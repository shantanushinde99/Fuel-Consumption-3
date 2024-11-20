import streamlit as st
import numpy as np
import pickle as pk

# Load the models and scaler
lr_filename = "trained_model_lr.sav"
dt_filename = "trained_model_dt.sav"
rf_filename = "trained_model_rf.sav"
scaler_filename = "scaled_data.sav"

# Load scaler and models
loaded_scaler = pk.load(open(scaler_filename, "rb"))
loaded_lr = pk.load(open(lr_filename, "rb"))
loaded_dt = pk.load(open(dt_filename, "rb"))
loaded_rf = pk.load(open(rf_filename, "rb"))

# Function to convert the input values
def input_converter(inp):
    vcl = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 
           'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 
           'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 
           'Pickup truck: Standard']
    trans = ['AV', 'AM', 'M', 'AS', 'A']
    fuel = ["D", "E", "X", "Z"]
    lst = []

    for i in range(6):
        if isinstance(inp[i], str):
            if inp[i] in vcl:
                lst.append(vcl.index(inp[i]))
            elif inp[i] in trans:
                lst.append(trans.index(inp[i]))
            elif inp[i] in fuel:
                lst.extend([1 if fuel.index(inp[i]) == j else 0 for j in range(4)])  # One-hot encoding for fuel types
                break
        else:
            lst.append(inp[i])

    arr = np.asarray(lst).reshape(1, -1)
    arr = loaded_scaler.transform(arr)
    return arr

# Streamlit app starts here
st.title('Fuel Consumption Prediction')

vehicle_classes = ['Two-seater', 'Minicompact', 'Compact', 'Subcompact', 'Mid-size', 'Full-size', 
                   'SUV: Small', 'SUV: Standard', 'Minivan', 'Station wagon: Small', 
                   'Station wagon: Mid-size', 'Pickup truck: Small', 'Special purpose vehicle', 
                   'Pickup truck: Standard']
transmission_types = ['AV', 'AM', 'M', 'AS', 'A']
fuel_types = ["D", "E", "X", "Z"]

# User input
vehicle_class = st.selectbox("Vehicle Class", vehicle_classes)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, format="%.1f")
cylinders = st.number_input("Cylinders", min_value=1, step=1)
transmission = st.selectbox("Transmission", transmission_types)
co2_rating = st.number_input("CO2 Rating", min_value=1, step=1)
fuel_type = st.selectbox("Fuel Type", fuel_types)

# Model selection
model_choice = st.selectbox("Select Model", ["Linear Regression", "Decision Tree", "Random Forest"])

# Button to predict
if st.button('Predict'):
    input_data = input_converter([vehicle_class, engine_size, cylinders, transmission, co2_rating, fuel_type])
    
    # Make predictions based on the selected model
    if model_choice == "Linear Regression":
        prediction = loaded_lr.predict(input_data)
    elif model_choice == "Decision Tree":
        prediction = loaded_dt.predict(input_data)
    elif model_choice == "Random Forest":
        prediction = loaded_rf.predict(input_data)
    
    # Display result
    st.success(f"Predicted Fuel Consumption: {round(prediction[0], 2)} L/100 km")
