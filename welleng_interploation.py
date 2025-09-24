import streamlit as st
import pandas as pd
from welleng.survey import Survey, interpolate_survey

st.title("Directional Survey Interpolation App")

# File uploader
uploaded_file = st.file_uploader("Upload your survey Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read Excel file
    df = pd.read_excel(uploaded_file)

    st.subheader("Uploaded Data")
    st.dataframe(df)

    # Prepare data
    md = [0] + df["MD"].tolist()
    inc = [0] + df["INC"].tolist()
    azi = [0] + df["AZI"].tolist()

    # Build survey
    survey = Survey(md, inc, azi, method="mincurv")

    # User input: interpolation step
    step = st.number_input("Interpolation Step (ft/m)", min_value=1, value=10, step=1)

    # Interpolate
    survey_interp = interpolate_survey(survey, step=step)

    # Create result DataFrame
    result = pd.DataFrame({
        "MD": survey_interp.md,
        "Inclination (deg)": survey_interp.inc_deg,
        "Azimuth (deg)": survey_interp.azi_mag_deg,
        "Latitude": survey_interp.n,
        "Departure": survey_interp.e,
        "TVD": survey_interp.tvd,
        "Vertical Section": survey_interp.vertical_section,
        "DLS": survey_interp.dls,
    }).round(2)

    st.subheader("Interpolated Survey")
    st.dataframe(result)

    # Download option
    st.download_button(
        label="Download Results as CSV",
        data=result.to_csv(index=False),
        file_name="interpolated_survey.csv",
        mime="text/csv"
    )
