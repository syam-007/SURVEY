import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from welleng.survey import Survey
import io

# --- Page setup ---
st.set_page_config(page_title="Wellpath Survey", layout="wide")
st.title("📊 Wellpath Survey Analysis")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload Survey Excel File", type=["xlsx"])

if uploaded_file:
    # --- Load survey data ---
    df = pd.read_excel(uploaded_file)

    md = [0] + df["MD"].tolist()
    inc = [0] + df["INC"].tolist()
    azi = [0] + df["AZI"].tolist()

    # --- Create Survey object ---
    survey = Survey(md, inc, azi, method="mincurv")
    survey.set_vertical_section(85.28, deg=True)

    closure_distance = []
    closure_direction_deg = []

    for i in range(len(survey.n)):
        delta_n = survey.n[i] - survey.n[0]
        delta_e = survey.e[i] - survey.e[0]
        dist = np.sqrt(delta_n ** 2 + delta_e ** 2)
        direction = np.degrees(np.arctan2(delta_e, delta_n)) % 360

        closure_distance.append(dist)
        closure_direction_deg.append(direction)

    # --- Results DataFrame ---
    result = pd.DataFrame({
        "MD": survey.md,
        "Inclination (deg)": survey.inc_deg,
        "Azimuth (deg)": survey.azi_grid_deg,
        "Latitude": survey.n,
        "Departure": survey.e,
        "TVD": survey.tvd,
        "Vertical Section": survey.vertical_section,
        "DLS": survey.dls,
        "Closure Distance": closure_distance,
        "Closure Direction (deg)": closure_direction_deg,
    }).round(2)

    # --- Display results ---
    st.subheader("📋 Survey Results")
    st.dataframe(result, use_container_width=True)

    # --- Download Excel ---
    excel_buffer = io.BytesIO()
    result.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Results as Excel",
        data=excel_buffer,
        file_name="survey_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --- Final values ---
    st.markdown(f"""
    ### 📌 Final Closure
    - **Distance:** {closure_distance[-1]:.2f} m  
    - **Direction:** {closure_direction_deg[-1]:.2f}°
    """)

    # --- 3D Wellpath Plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(result["Departure"], result["Latitude"], -result["TVD"], marker="o")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("TVD")
    ax.set_title("3D Wellpath")
    st.pyplot(fig)

else:
    st.info("👆 Upload an Excel file to start the analysis.")
