import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from welleng.survey import Survey, interpolate_survey
import welleng as we
import io
import plotly.graph_objects as go

# --- Page setup ---
st.set_page_config(page_title="Wellpath Survey Toolkit", layout="wide")

# =====================================================
# SECTION-1: Survey Calculation
# =====================================================
st.title("Section-1 🔖 Survey Calculation")

uploaded_file_1 = st.file_uploader("Upload Survey Excel File (Section-1)", type=["xlsx"], key="sec1")

if uploaded_file_1:
    df1 = pd.read_excel(uploaded_file_1)

    md = [0] + df1["MD"].tolist()
    inc = [0] + df1["INC"].tolist()
    azi = [0] + df1["AZI"].tolist()

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

    st.subheader("📋 Survey Results")
    st.dataframe(result, use_container_width=True)

    excel_buffer = io.BytesIO()
    result.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_buffer.seek(0)
    st.download_button("📥 Download Results as Excel", data=excel_buffer,
                       file_name="survey_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown(f"""
    ### 📌 Final Closure
    - **Distance:** {closure_distance[-1]:.2f} m  
    - **Direction:** {closure_direction_deg[-1]:.2f}°
    """)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(result["Departure"], result["Latitude"], -result["TVD"], marker="o")
    ax.set_xlabel("Easting"); ax.set_ylabel("Northing"); ax.set_zlabel("TVD")
    ax.set_title("3D Wellpath")
    st.pyplot(fig)
else:
    st.info("👆 Upload an Excel file in Section-1 to start the analysis.")


# =====================================================
# SECTION-2: Survey Interpolation
# =====================================================
st.title("Section-2 🔖 Survey Interpolation")

uploaded_file_2 = st.file_uploader("Upload Survey Excel File (Section-2)", type=["xlsx"], key="sec2")

if uploaded_file_2:
    df2 = pd.read_excel(uploaded_file_2)
    st.subheader("Uploaded Data (Section-2)")
    st.dataframe(df2)

    md = [0] + df2["MD"].tolist()
    inc = [0] + df2["INC"].tolist()
    azi = [0] + df2["AZI"].tolist()

    survey2 = Survey(md, inc, azi, method="mincurv")
    step = st.number_input("Interpolation Step (ft/m)", min_value=1, value=10, step=1, key="interp_step")
    survey_interp = interpolate_survey(survey2, step=step)

    interp_result = pd.DataFrame({
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
    st.dataframe(interp_result)

    st.download_button("📥 Download Interpolated Survey (CSV)",
                       data=interp_result.to_csv(index=False),
                       file_name="interpolated_survey.csv", mime="text/csv")
else:
    st.info("👆 Upload an Excel file in Section-2 to run interpolation.")


# =====================================================
# SECTION-3: Survey Comparison
# =====================================================
st.title("Section-3 🔖 Well Survey Comparison Tool")

ref_file = st.file_uploader("Choose Reference Survey", type=['csv', 'xlsx'], key="ref")
cmp_file = st.file_uploader("Choose Comparative Survey", type=['csv', 'xlsx'], key="cmp")
step_cmp = st.number_input("Interpolation Step (MD units)", min_value=1, value=10, step=1, key="cmp_step")

def load_survey(file):
    if file is None: return None
    if str(file.name).lower().endswith('.csv'): df = pd.read_csv(file)
    else: df = pd.read_excel(file)
    if not {'MD','INC','AZI'}.issubset(df.columns):
        st.error("File must contain columns: MD, INC, AZI"); return None
    return df

def manual_interpolation(df_ref, df_cmp, step):
    md_min = max(df_ref['MD'].min(), df_cmp['MD'].min())
    md_max = min(df_ref['MD'].max(), df_cmp['MD'].max())
    md_common = np.arange(md_min, md_max+step, step)
    inc_ref = np.interp(md_common, df_ref['MD'], df_ref['INC'])
    azi_ref = np.interp(md_common, df_ref['MD'], df_ref['AZI'])
    inc_cmp = np.interp(md_common, df_cmp['MD'], df_cmp['INC'])
    azi_cmp = np.interp(md_common, df_cmp['MD'], df_cmp['AZI'])
    return md_common, inc_ref, azi_ref, inc_cmp, azi_cmp

def create_interpolated_survey(md, inc, azi):
    return we.survey.Survey(md=md, inc=inc, azi=azi, start_xyz=[0,0,0])

if ref_file and cmp_file:
    df_ref = load_survey(ref_file); df_cmp = load_survey(cmp_file)
    if df_ref is not None and df_cmp is not None:
        md_common, inc_ref, azi_ref, inc_cmp, azi_cmp = manual_interpolation(df_ref, df_cmp, step_cmp)
        survey_ref = create_interpolated_survey(md_common, inc_ref, azi_ref)
        survey_cmp = create_interpolated_survey(md_common, inc_cmp, azi_cmp)

        dx, dy, dz = survey_ref.n-survey_cmp.n, survey_ref.e-survey_cmp.e, survey_ref.tvd-survey_cmp.tvd
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)

        output = pd.DataFrame({
            'MD': survey_ref.md,
            'INC_ref': survey_ref.inc_deg, 'AZI_ref': survey_ref.azi_grid_deg,
            'INC_cmp': survey_cmp.inc_deg, 'AZI_cmp': survey_cmp.azi_grid_deg,
            'Delta_N': dx, 'Delta_E': dy, 'Delta_TVD': dz, 'Displacement': displacement
        })
        st.success(f"Interpolation complete! {len(md_common)} points compared.")
        st.dataframe(output)
        st.download_button("📥 Download Comparison CSV",
                           data=output.to_csv(index=False), file_name="survey_comparison.csv", mime="text/csv")

        tab1, tab2, tab3 = st.tabs(["3D Wellbore Path","Displacement vs MD","Delta INC & AZI vs MD"])
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(x=survey_ref.e,y=survey_ref.n,z=survey_ref.tvd,mode='lines',name='Reference',line=dict(color='blue')))
            fig.add_trace(go.Scatter3d(x=survey_cmp.e,y=survey_cmp.n,z=survey_cmp.tvd,mode='lines',name='Comparative',line=dict(color='red')))
            fig.update_layout(scene=dict(xaxis_title='East',yaxis_title='North',zaxis_title='TVD',zaxis=dict(autorange="reversed")),height=800)
            st.plotly_chart(fig,use_container_width=True)
        with tab2:
            st.plotly_chart(go.Figure(go.Scatter(x=output['MD'],y=output['Displacement'],mode='lines+markers')),use_container_width=True)
        with tab3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=survey_ref.md,y=survey_ref.inc_deg-survey_cmp.inc_deg,name="ΔINC"))
            fig3.add_trace(go.Scatter(x=survey_ref.md,y=((survey_ref.azi_grid_deg-survey_cmp.azi_grid_deg)+180)%360-180,name="ΔAZI"))
            st.plotly_chart(fig3,use_container_width=True)
else:
    st.info("👆 Upload both Reference and Comparative survey files in Section-3 to run comparison.")


# =====================================================
# SECTION-4: Advanced Survey Comparison with Offsets
# =====================================================
st.title("Section-4 🔖 Advanced Survey Comparison with Offsets")

ref_file4 = st.file_uploader("Choose Reference Survey", type=['csv', 'xlsx'], key="ref4")
cmp_file4 = st.file_uploader("Choose Comparative Survey", type=['csv', 'xlsx'], key="cmp4")
step4 = st.number_input("Interpolation Step (MD units)", min_value=1, value=10, step=1, key="cmp_step4")

if ref_file4 and cmp_file4:
    df_ref = load_survey(ref_file4)
    df_cmp = load_survey(cmp_file4)

    if df_ref is not None and df_cmp is not None:
        # Interpolate surveys
        md_common, inc_ref, azi_ref, inc_cmp, azi_cmp = manual_interpolation(df_ref, df_cmp, step4)
        survey_ref = create_interpolated_survey(md_common, inc_ref, azi_ref)
        survey_cmp = create_interpolated_survey(md_common, inc_cmp, azi_cmp)

        # --- session_state for adjustment history ---
        if "adjusted_survey" not in st.session_state: st.session_state.adjusted_survey = None
        if "history" not in st.session_state: st.session_state.history = []
        if "future" not in st.session_state: st.session_state.future = []

        # --- Tabs ---
        tab1, tab2, tab3 = st.tabs(["3D Wellbore Path + Offsets", "Displacement vs MD", "Delta INC & AZI vs MD"])

        # ---------------- Tab 1 ----------------
        with tab1:
            col1, col2 = st.columns([2, 1])

            # Controls
            with col2:
                st.markdown("### Adjust Comparative Survey with Offsets")

                md_start = st.number_input("Start MD for offset", value=float(survey_cmp.md.min()), step=10.0)
                md_end = st.number_input("End MD for offset", value=float(survey_cmp.md.max()), step=10.0)

                x_offset = st.number_input("Adjust Easting (X)", value=0.0, step=1.0, key="x_offset4")
                y_offset = st.number_input("Adjust Northing (Y)", value=0.0, step=1.0, key="y_offset4")
                z_offset = st.number_input("Adjust TVD (Z)", value=0.0, step=1.0, key="z_offset4")

                # --- Apply Offsets ---
                if st.button("Apply Offsets"):
                    if st.session_state.adjusted_survey is not None:
                        base_e = st.session_state.adjusted_survey["East_adj"].to_numpy()
                        base_n = st.session_state.adjusted_survey["North_adj"].to_numpy()
                        base_tvd = st.session_state.adjusted_survey["TVD_adj"].to_numpy()
                        st.session_state.history.append(st.session_state.adjusted_survey.copy())
                    else:
                        base_e, base_n, base_tvd = survey_cmp.e.copy(), survey_cmp.n.copy(), survey_cmp.tvd.copy()

                    st.session_state.future = []  # clear redo
                    mask = (survey_cmp.md >= md_start) & (survey_cmp.md <= md_end)
                    base_e[mask] += x_offset
                    base_n[mask] += y_offset
                    base_tvd[mask] += z_offset

                    adjusted_output = pd.DataFrame({
                        "MD": survey_cmp.md,
                        "North_adj": base_n,
                        "East_adj": base_e,
                        "TVD_adj": base_tvd
                    })
                    st.session_state.adjusted_survey = adjusted_output
                    st.success("Offsets applied!")
                    st.dataframe(adjusted_output, height=400)
                    st.download_button("Download Adjusted Survey",
                                       data=adjusted_output.to_csv(index=False).encode("utf-8"),
                                       file_name="adjusted_survey.csv", mime="text/csv")

                # --- Undo ---
                if st.button("Undo Last Change"):
                    if st.session_state.history:
                        st.session_state.future.append(st.session_state.adjusted_survey.copy())
                        st.session_state.adjusted_survey = st.session_state.history.pop()
                        st.success("Undo successful.")
                    else:
                        st.warning("No change to undo.")

                # --- Redo ---
                if st.button("Redo (Forward)"):
                    if st.session_state.future:
                        st.session_state.history.append(st.session_state.adjusted_survey.copy())
                        st.session_state.adjusted_survey = st.session_state.future.pop()
                        st.success("Redo successful.")
                    else:
                        st.warning("No redo available.")

                # --- Reset ---
                if st.button("Reset to Original"):
                    st.session_state.adjusted_survey = None
                    st.session_state.history, st.session_state.future = [], []
                    st.success("Reset to original survey.")

                # --- Recalculate INC/AZI ---
                if st.button("Recalculate MD/INC/AZI from Adjusted Path"):
                    if st.session_state.adjusted_survey is not None:
                        df_adj = st.session_state.adjusted_survey
                        n_adj, e_adj, tvd_adj = df_adj["North_adj"].to_numpy(), df_adj["East_adj"].to_numpy(), df_adj["TVD_adj"].to_numpy()
                        md_vals = survey_cmp.md
                        dN, dE, dTVD = np.gradient(n_adj, md_vals), np.gradient(e_adj, md_vals), np.gradient(tvd_adj, md_vals)
                        inc_adj = np.degrees(np.arctan2(np.sqrt(dN**2 + dE**2), dTVD))
                        azi_adj = (np.degrees(np.arctan2(dE, dN)) + 360) % 360

                        recalculated_output = pd.DataFrame({
                            "MD": md_vals,
                            "North": n_adj,
                            "East": e_adj,
                            "TVD": tvd_adj,
                            "INC_recalc": inc_adj,
                            "AZI_recalc": azi_adj
                        })
                        st.success("Recalculated INC/AZI from adjusted survey.")
                        st.dataframe(recalculated_output, height=400)
                        st.download_button("Download Recalculated Survey",
                                           data=recalculated_output.to_csv(index=False).encode("utf-8"),
                                           file_name="recalculated_survey.csv", mime="text/csv")
                    else:
                        st.warning("Apply offsets first.")

                # --- Compare Adjusted vs Reference ---
                if st.button("Compare Adjusted vs Reference"):
                    if st.session_state.adjusted_survey is not None:
                        df_adj = st.session_state.adjusted_survey
                        survey_cmp_adj = we.survey.Survey(md=survey_cmp.md,
                                                          inc=survey_cmp.inc_deg,
                                                          azi=survey_cmp.azi_grid_deg,
                                                          start_xyz=[0,0,0])
                        survey_cmp_adj.n = df_adj["North_adj"].to_numpy()
                        survey_cmp_adj.e = df_adj["East_adj"].to_numpy()
                        survey_cmp_adj.tvd = df_adj["TVD_adj"].to_numpy()

                        dx, dy, dz = survey_ref.n - survey_cmp_adj.n, survey_ref.e - survey_cmp_adj.e, survey_ref.tvd - survey_cmp_adj.tvd
                        displacement = np.sqrt(dx**2 + dy**2 + dz**2)
                        compare_output = pd.DataFrame({
                            "MD": survey_ref.md,
                            "Delta_N": dx,
                            "Delta_E": dy,
                            "Delta_TVD": dz,
                            "Displacement": displacement
                        })
                        st.success("Comparison complete.")
                        st.dataframe(compare_output, height=400)
                        st.download_button("Download Comparison CSV",
                                           data=compare_output.to_csv(index=False).encode("utf-8"),
                                           file_name="adjusted_comparison.csv", mime="text/csv")
                    else:
                        st.warning("Apply offsets first.")

            # 3D Plot
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=survey_ref.e, y=survey_ref.n, z=survey_ref.tvd,
                                           mode="lines", name="Reference", line=dict(color="blue")))
                if st.session_state.adjusted_survey is not None:
                    fig.add_trace(go.Scatter3d(x=st.session_state.adjusted_survey["East_adj"],
                                               y=st.session_state.adjusted_survey["North_adj"],
                                               z=st.session_state.adjusted_survey["TVD_adj"],
                                               mode="lines", name="Comparative (Adjusted)", line=dict(color="red")))
                else:
                    fig.add_trace(go.Scatter3d(x=survey_cmp.e, y=survey_cmp.n, z=survey_cmp.tvd,
                                               mode="lines", name="Comparative", line=dict(color="red")))
                fig.update_layout(scene=dict(xaxis_title="East [m]", yaxis_title="North [m]", zaxis_title="TVD [m]",
                                             zaxis=dict(autorange="reversed"), aspectmode="cube"), height=800)
                st.plotly_chart(fig, use_container_width=True)

        # ---------------- Tab 2 ----------------
        with tab2:
            dx, dy, dz = survey_ref.n - survey_cmp.n, survey_ref.e - survey_cmp.e, survey_ref.tvd - survey_cmp.tvd
            displacement = np.sqrt(dx**2 + dy**2 + dz**2)
            fig2 = go.Figure(go.Scatter(x=survey_ref.md, y=displacement, mode="lines+markers"))
            fig2.update_layout(title="3D Displacement vs MD", xaxis_title="MD [m]", yaxis_title="Displacement [m]")
            st.plotly_chart(fig2, use_container_width=True)

        # ---------------- Tab 3 ----------------
        with tab3:
            delta_inc = survey_ref.inc_deg - survey_cmp.inc_deg
            delta_azi = ((survey_ref.azi_grid_deg - survey_cmp.azi_grid_deg) + 180) % 360 - 180
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=survey_ref.md, y=delta_inc, name="ΔINC"))
            fig3.add_trace(go.Scatter(x=survey_ref.md, y=delta_azi, name="ΔAZI"))
            fig3.update_layout(title="Delta INC & AZI vs MD", xaxis_title="MD [m]", yaxis_title="Degrees")
            st.plotly_chart(fig3, use_container_width=True)
# =====================================================
# SECTION-5: Well Survey Extrapolation Tool
# =====================================================
st.title("Section-5 🔖 Well Survey Extrapolation Tool")

uploaded_file5 = st.file_uploader("Upload Survey File (CSV/Excel)", type=['csv','xlsx'], key="sec5")

if uploaded_file5 is not None:
    # --- Load Data ---
    if str(uploaded_file5.name).lower().endswith('.csv'):
        df = pd.read_csv(uploaded_file5)
    else:
        df = pd.read_excel(uploaded_file5)

    # Ensure required columns exist
    column_mapping = {
        'MD': ['MD','md','Measured Depth','MeasuredDepth'],
        'INC': ['INC','inc','Inclination','incl'],
        'AZI': ['AZI','azi','Azimuth','azimuth']
    }
    for required_col, alternatives in column_mapping.items():
        found = False
        for alt in alternatives:
            if alt in df.columns:
                df = df.rename(columns={alt: required_col})
                found = True
                break
        if not found:
            st.error(f"Could not find {required_col} column. Available: {list(df.columns)}")
            st.stop()

    # --- Sidebar Controls ---
    st.sidebar.header("Section-5 Settings")
    extrapolation_length = st.sidebar.number_input("Extrapolation Length (m)", 0, 10000, 200, 50, key="extrap_len5")
    extrapolation_step = st.sidebar.number_input("Extrapolation Step (m)", 1, 50, 10, 1, key="extrap_step5")
    interpolation_step = st.sidebar.number_input("Interpolation Step (m)", 1, 50, 10, 1, key="interp_step5")
    extrapolation_method = st.sidebar.selectbox("Extrapolation Method",
                                                ["Constant","Linear Trend","Curve Fit"],
                                                key="extrap_method5")

    # --- Process Data ---
    df = df.sort_values('MD').drop_duplicates(subset='MD')
    survey_original = we.survey.Survey(md=df['MD'].tolist(),
                                       inc=df['INC'].tolist(),
                                       azi=df['AZI'].tolist())
    survey_interp = survey_original.interpolate_survey(step=interpolation_step)

    # --- Extrapolation Function ---
    def extrapolate_survey(survey, length=100, step=10, method="Constant"):
        if length <= 0: return survey, np.array([])
        last_md = survey.md[-1]
        extrapolation_mds = np.arange(last_md+step, last_md+length+step, step)
        if method=="Constant":
            extrap_incs = [survey.inc_deg[-1]]*len(extrapolation_mds)
            extrap_azis = [survey.azi_grid_deg[-1]]*len(extrapolation_mds)
        elif method=="Linear Trend":
            n_points=min(5,len(survey.inc_deg))
            if n_points>1:
                inc_trend=np.polyfit(survey.md[-n_points:],survey.inc_deg[-n_points:],1)
                azi_trend=np.polyfit(survey.md[-n_points:],survey.azi_grid_deg[-n_points:],1)
                extrap_incs=np.polyval(inc_trend,extrapolation_mds)
                extrap_azis=np.polyval(azi_trend,extrapolation_mds)
            else:
                extrap_incs=[survey.inc_deg[-1]]*len(extrapolation_mds)
                extrap_azis=[survey.azi_grid_deg[-1]]*len(extrapolation_mds)
        elif method=="Curve Fit":
            n_points=min(10,len(survey.inc_deg))
            if n_points>2:
                inc_fit=np.polyfit(survey.md[-n_points:],survey.inc_deg[-n_points:],2)
                azi_fit=np.polyfit(survey.md[-n_points:],survey.azi_grid_deg[-n_points:],2)
                extrap_incs=np.polyval(inc_fit,extrapolation_mds)
                extrap_azis=np.polyval(azi_fit,extrapolation_mds)
            else:
                extrap_incs=[survey.inc_deg[-1]]*len(extrapolation_mds)
                extrap_azis=[survey.azi_grid_deg[-1]]*len(extrapolation_mds)
        combined_md=np.concatenate([survey.md,extrapolation_mds])
        combined_inc=np.concatenate([survey.inc_deg,extrap_incs])
        combined_azi=np.concatenate([survey.azi_grid_deg,extrap_azis])
        extrapolated_survey=we.survey.Survey(md=combined_md.tolist(),
                                             inc=combined_inc.tolist(),
                                             azi=combined_azi.tolist())
        return extrapolated_survey, extrapolation_mds

    survey_extrapolated, extrap_points = extrapolate_survey(survey_interp,
                                                            extrapolation_length,
                                                            extrapolation_step,
                                                            extrapolation_method)

    st.success(f"✅ Loaded {len(df)} survey points")
    if extrapolation_length>0:
        st.info(f"📈 Added {len(extrap_points)} extrapolated points")

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["3D View","2D Views","Survey Data"])

    with tab1:
        st.subheader("3D Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=survey_original.e,y=survey_original.n,z=survey_original.tvd,
                                   mode='markers',marker=dict(size=4,color='blue'),name='Original'))
        fig.add_trace(go.Scatter3d(x=survey_interp.e,y=survey_interp.n,z=survey_interp.tvd,
                                   mode='lines',line=dict(color='green',width=3),name='Interpolated'))
        if extrapolation_length>0:
            idx=len(survey_interp.md)
            fig.add_trace(go.Scatter3d(x=survey_extrapolated.e[idx:],y=survey_extrapolated.n[idx:],z=survey_extrapolated.tvd[idx:],
                                       mode='lines',line=dict(color='red',width=3,dash='dash'),name='Extrapolated'))
        fig.update_layout(scene=dict(xaxis_title='East',yaxis_title='North',zaxis_title='TVD',
                                     zaxis=dict(autorange="reversed"),aspectmode="cube"),
                          height=700)
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        st.subheader("2D Views")
        col1,col2=st.columns(2)
        with col1:
            st.write("Plan View")
            fig_plan=go.Figure()
            fig_plan.add_trace(go.Scatter(x=survey_interp.e,y=survey_interp.n,mode='lines',name='Interpolated'))
            if extrapolation_length>0:
                idx=len(survey_interp.md)
                fig_plan.add_trace(go.Scatter(x=survey_extrapolated.e[idx:],y=survey_extrapolated.n[idx:],mode='lines',name='Extrapolated'))
            fig_plan.add_trace(go.Scatter(x=survey_original.e,y=survey_original.n,mode='markers',name='Original'))
            st.plotly_chart(fig_plan,use_container_width=True)
        with col2:
            st.write("Vertical Section")
            horiz=np.sqrt(survey_interp.e**2+survey_interp.n**2)
            fig_vert=go.Figure()
            fig_vert.add_trace(go.Scatter(x=horiz,y=survey_interp.tvd,mode='lines',name='Interpolated'))
            if extrapolation_length>0:
                idx=len(survey_interp.md)
                horiz_ex=np.sqrt(np.array(survey_extrapolated.e[idx:])**2+np.array(survey_extrapolated.n[idx:])**2)
                fig_vert.add_trace(go.Scatter(x=horiz_ex,y=survey_extrapolated.tvd[idx:],mode='lines',name='Extrapolated'))
            fig_vert.add_trace(go.Scatter(x=np.sqrt(survey_original.e**2+survey_original.n**2),y=survey_original.tvd,
                                          mode='markers',name='Original'))
            fig_vert.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_vert,use_container_width=True)

    with tab3:
        st.subheader("Survey Data Table")
        out=[]
        for i in range(len(survey_extrapolated.md)):
            out.append({
                "MD": survey_extrapolated.md[i],
                "INC": survey_extrapolated.inc_deg[i],
                "AZI": survey_extrapolated.azi_grid_deg[i],
                "North": survey_extrapolated.n[i],
                "East": survey_extrapolated.e[i],
                "TVD": survey_extrapolated.tvd[i],
                "Type": "Original" if i < len(survey_original.md) else "Extrapolated"
            })
        df_out=pd.DataFrame(out).round(2)
        st.dataframe(df_out,height=400)
        st.download_button("📥 Download Survey Data",
                           data=df_out.to_csv(index=False).encode('utf-8'),
                           file_name="section5_extrapolated_survey.csv",mime="text/csv")

else:
    st.info("👆 Upload a survey file in Section-5 to run extrapolation.")
