import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title("Dry Bean Classification App")

model = joblib.load("dry_bean_classifier.pkl")
scaler = joblib.load("scaler.pkl")


area = st.slider("Area", min_value=100.0, max_value=50000.0, step=0.1)
perimeter = st.slider("Perimeter", min_value=100.0, max_value=3000.0, step=0.01)
major_axis_length = st.slider("MajorAxisLength", min_value=10.0, max_value=1000.0, step=0.01)
minor_axis_length = st.slider("MinorAxisLength", min_value=10.0, max_value=1000.0, step=0.01)
aspect_ratio = st.number_input ("AspectRatio", min_value=0.0, max_value=10.0, step=0.01)
eccentricity = st.number_input("Eccentricity", min_value=0.0, max_value=1.0, step=0.01)            
convex_area = st.slider("ConvexArea", min_value=1000.0, max_value=50000.0, step=0.01)
equivalent_diameter = st.slider("EquivDiameter", min_value=100.00, max_value=1000.00, step=0.01)
extent = st.number_input("Extent", min_value=0.0, max_value=1.0, step=0.01)
solidity = st.number_input("Solidity", min_value=0.0, max_value=1.0, step=0.01)
roundness = st.number_input("Roundness", min_value=0.0, max_value=1.0, step=0.01)
compactness = st.number_input("Compactness", min_value=0.0, max_value=1.0, step=0.01)
shape_factor1 = st.number_input("ShapeFactor1", min_value=0.0001, max_value=1.0, step=0.01)
shape_factor2 = st.number_input("ShapeFactor2", min_value=0.0001, max_value=1.0, step=0.01)       
shape_factor3 = st.number_input("ShapeFactor3", min_value=0.0, max_value=1.0, step=0.01)
shape_factor4 = st.number_input("ShapeFactor4", min_value=0.0, max_value=1.0, step=0.01)

st.write("Please enter the values for the features above and click the 'Predict' button to classify the dry bean type.")    

if st.button("Predict", key="predict_button"):
    input_data = np.array([[area, perimeter, major_axis_length, minor_axis_length, aspect_ratio, eccentricity,
                            convex_area, equivalent_diameter, extent, solidity, roundness,
                            compactness, shape_factor1, shape_factor2, shape_factor3, shape_factor4]])
    
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    
    st.write(f"The predicted dry bean type is: {prediction[0]}")        