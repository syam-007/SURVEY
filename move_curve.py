import streamlit as st
import pandas as pd
import welleng as we
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Well Survey Comparison", layout="wide")
st.title("Well Survey Comparison Tool")

# --- Upload reference survey ---
st.subheader("Upload Reference Survey (CSV or Excel)")
ref_file = st.file_uploader("Choose Reference Survey", type=['csv', 'xlsx'])

# --- Upload comparative survey ---
st.subheader("Upload Comparative Survey (CSV or Excel)")
cmp_file = st.file_uploader("Choose Comparative Survey", type=['csv', 'xlsx'])

# --- Interpolation step ---
step = st.number_input("Interpolation Step (MD units)", min_value=1, value=10, step=1)

def load_survey(file):
    if file is None:
        return None
    if str(file.name).lower().endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    if not {'MD', 'INC', 'AZI'}.issubset(df.columns):
        st.error("File must contain columns: MD, INC, AZI")
        return None
    return df

def manual_interpolation(df_ref, df_cmp, step):
    md_min = max(df_ref['MD'].min(), df_cmp['MD'].min())
    md_max = min(df_ref['MD'].max(), df_cmp['MD'].max())
    md_common = np.arange(md_min, md_max + step, step)
    md_common = np.unique(np.sort(md_common))

    inc_ref = np.interp(md_common, df_ref['MD'], df_ref['INC'])
    azi_ref = np.interp(md_common, df_ref['MD'], df_ref['AZI'])
    inc_cmp = np.interp(md_common, df_cmp['MD'], df_cmp['INC'])
    azi_cmp = np.interp(md_common, df_cmp['MD'], df_cmp['AZI'])

    return md_common, inc_ref, azi_ref, inc_cmp, azi_cmp

def create_interpolated_survey(md, inc, azi):
    return we.survey.Survey(md=md, inc=inc, azi=azi, start_xyz=[0,0,0])

if ref_file and cmp_file:

    df_ref = load_survey(ref_file)
    df_cmp = load_survey(cmp_file)

    if df_ref is not None and df_cmp is not None:
        # Interpolate surveys
        md_common, inc_ref, azi_ref, inc_cmp, azi_cmp = manual_interpolation(df_ref, df_cmp, step)

        survey_ref = create_interpolated_survey(md_common, inc_ref, azi_ref)
        survey_cmp = create_interpolated_survey(md_common, inc_cmp, azi_cmp)

        # Calculate deltas
        delta_inc = survey_ref.inc_deg - survey_cmp.inc_deg
        delta_azi = ((survey_ref.azi_grid_deg - survey_cmp.azi_grid_deg) + 180) % 360 - 180
        dx = survey_ref.n - survey_cmp.n
        dy = survey_ref.e - survey_cmp.e
        dz = survey_ref.tvd - survey_cmp.tvd
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)

        # Prepare DataFrame
        output = pd.DataFrame({
            'MD': survey_ref.md,
            'INC_ref': survey_ref.inc_deg,
            'AZI_ref': survey_ref.azi_grid_deg,
            'North_ref': survey_ref.n,
            'East_ref': survey_ref.e,
            'TVD_ref': survey_ref.tvd,
            'INC_cmp': survey_cmp.inc_deg,
            'AZI_cmp': survey_cmp.azi_grid_deg,
            'North_cmp': survey_cmp.n,
            'East_cmp': survey_cmp.e,
            'TVD_cmp': survey_cmp.tvd,
            'Delta_INC': delta_inc,
            'Delta_AZI': abs(delta_azi),
            'Delta_N': dx,
            'Delta_E': dy,
            'Delta_TVD': dz,
            'Displacement': displacement
        })

        st.success(f"Interpolation complete! {len(md_common)} points compared from MD {md_common[0]} to {md_common[-1]}.")
        st.dataframe(output)

        # Download CSV
        csv = output.to_csv(index=False).encode('utf-8')
        st.download_button("Download Comparison CSV", data=csv, file_name="survey_comparison.csv", mime="text/csv")

        # --- Create tabs for graphs ---
        tab1, tab2, tab3 = st.tabs(["3D Wellbore Path", "Displacement vs MD", "Delta INC & AZI vs MD"])

        # 3D Wellbore Path

        with tab1:
            # Define columns: left (graph), right (controls)
            col1, col2 = st.columns([2, 1])
            if "adjusted_survey" not in st.session_state:
                st.session_state.adjusted_survey = None
            if "history" not in st.session_state:
                st.session_state.history = []

            if "future" not in st.session_state:
                st.session_state.future = []

            # Offsets must be defined first
            with col2:
                st.markdown("### Adjust Comparative Survey with Offsets")

                md_start = st.number_input("Start MD for offset", value=float(survey_cmp.md.min()), step=10.0)
                md_end = st.number_input("End MD for offset", value=float(survey_cmp.md.max()), step=10.0)

                x_offset = st.number_input("Adjust Easting (X)", value=0.0, step=1.0, key="x_offset")
                y_offset = st.number_input("Adjust Northing (Y)", value=0.0, step=1.0, key="y_offset")
                z_offset = st.number_input("Adjust TVD (Z)", value=0.0, step=1.0, key="z_offset")
                if st.button("Undo Last Change"):
                    if st.session_state.history:
                        st.session_state.future.append(st.session_state.adjusted_survey.copy())
                        st.session_state.adjusted_survey = st.session_state.history.pop()
                        st.success("Last change undone.")
                        st.dataframe(st.session_state.adjusted_survey, height=400)
                    else:
                        st.warning("No previous change to undo.")

                    # Redo button
                if st.button("Redo (Forward)"):
                    if st.session_state.future:
                        st.session_state.history.append(st.session_state.adjusted_survey.copy())
                        st.session_state.adjusted_survey = st.session_state.future.pop()
                        st.success("Redo applied (moved forward).")
                        st.dataframe(st.session_state.adjusted_survey, height=400)
                    else:
                        st.warning("No forward state to redo.")

                    # Reset button
                if st.button("Reset to Original"):
                    st.session_state.adjusted_survey = None
                    st.session_state.history = []
                    st.session_state.future = []
                    st.success("Survey reset to original.")
                if st.button("Recalculate MD/INC/AZI from Adjusted Path"):
                    if st.session_state.adjusted_survey is not None:
                        df_adj = st.session_state.adjusted_survey

                        # Extract coords
                        n_adj = df_adj["North_adj"].to_numpy()
                        e_adj = df_adj["East_adj"].to_numpy()
                        tvd_adj = df_adj["TVD_adj"].to_numpy()
                        md_vals = survey_cmp.md

                        # Gradients relative to MD
                        dN = np.gradient(n_adj, md_vals)
                        dE = np.gradient(e_adj, md_vals)
                        dTVD = np.gradient(tvd_adj, md_vals)

                        # Compute INC & AZI
                        inc_adj = np.degrees(np.arctan2(np.sqrt(dN ** 2 + dE ** 2), dTVD))
                        azi_adj = (np.degrees(np.arctan2(dE, dN)) + 360) % 360

                        recalculated_output = pd.DataFrame({
                            "MD": md_vals,
                            "North": n_adj,
                            "East": e_adj,
                            "TVD": tvd_adj,
                            "INC_recalc": inc_adj,
                            "AZI_recalc": azi_adj
                        })

                        st.success("Recalculated MD, INC, and AZI from adjusted survey!")
                        st.dataframe(recalculated_output, height=400)

                        # Download
                        csv_recalc = recalculated_output.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Recalculated Survey", data=csv_recalc,
                                           file_name="recalculated_survey.csv", mime="text/csv")
                    else:
                        st.warning("No adjusted survey available. Apply offsets first.")
                if st.button("Compare"):
                    if st.session_state.adjusted_survey is not None:
                        df_adj = st.session_state.adjusted_survey

                        # Build adjusted survey using welleng
                        survey_cmp_adj = we.survey.Survey(
                            md=survey_cmp.md,
                            inc=np.interp(survey_cmp.md, survey_cmp.md, survey_cmp.inc_deg),  # start with same INC
                            azi=np.interp(survey_cmp.md, survey_cmp.md, survey_cmp.azi_grid_deg),  # same AZI
                            start_xyz=[0, 0, 0]
                        )

                        # Replace coords with adjusted values
                        survey_cmp_adj.n = df_adj["North_adj"].to_numpy()
                        survey_cmp_adj.e = df_adj["East_adj"].to_numpy()
                        survey_cmp_adj.tvd = df_adj["TVD_adj"].to_numpy()

                        # Recalculate deltas vs reference
                        delta_inc = survey_ref.inc_deg - survey_cmp_adj.inc_deg
                        delta_azi = ((survey_ref.azi_grid_deg - survey_cmp_adj.azi_grid_deg) + 180) % 360 - 180
                        dx = survey_ref.n - survey_cmp_adj.n
                        dy = survey_ref.e - survey_cmp_adj.e
                        dz = survey_ref.tvd - survey_cmp_adj.tvd
                        displacement = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

                        compare_output = pd.DataFrame({
                            'MD': survey_ref.md,
                            'INC_ref': survey_ref.inc_deg,
                            'AZI_ref': survey_ref.azi_grid_deg,
                            'North_ref': survey_ref.n,
                            'East_ref': survey_ref.e,
                            'TVD_ref': survey_ref.tvd,
                            'INC_adj': survey_cmp_adj.inc_deg,
                            'AZI_adj': survey_cmp_adj.azi_grid_deg,
                            'North_adj': survey_cmp_adj.n,
                            'East_adj': survey_cmp_adj.e,
                            'TVD_adj': survey_cmp_adj.tvd,
                            'Delta_INC': delta_inc,
                            'Delta_AZI': abs(delta_azi),
                            'Delta_N': dx,
                            'Delta_E': dy,
                            'Delta_TVD': dz,
                            'Displacement': displacement

                        })

                        st.success("Comparison complete: Adjusted survey vs Reference survey")
                        st.dataframe(compare_output, height=400)

                        # Download results
                        csv_comp = compare_output.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Comparison CSV", data=csv_comp,
                                           file_name="adjusted_comparison.csv", mime="text/csv")
                    else:
                        st.warning("No adjusted survey found. Apply offsets first.")

                # Apply offsets
                if st.button("Apply Offsets"):
                    # Use adjusted or original as base
                    if st.session_state.adjusted_survey is not None:
                        base_e = st.session_state.adjusted_survey["East_adj"].to_numpy()
                        base_n = st.session_state.adjusted_survey["North_adj"].to_numpy()
                        base_tvd = st.session_state.adjusted_survey["TVD_adj"].to_numpy()
                    else:
                        base_e = survey_cmp.e.copy()
                        base_n = survey_cmp.n.copy()
                        base_tvd = survey_cmp.tvd.copy()

                    # Save current state into history for undo
                    if st.session_state.adjusted_survey is not None:
                        st.session_state.history.append(st.session_state.adjusted_survey.copy())

                    # Clear redo stack (new branch of changes)
                    st.session_state.future = []

                    # Apply offsets only in MD window
                    mask = (survey_cmp.md >= md_start) & (survey_cmp.md <= md_end)
                    base_e[mask] += x_offset
                    base_n[mask] += y_offset
                    base_tvd[mask] += z_offset



                    # Store new adjusted survey
                    adjusted_output = pd.DataFrame({
                        "MD": survey_cmp.md,
                        "North_adj": base_n,
                        "East_adj": base_e,
                        "TVD_adj": base_tvd
                    })
                    st.session_state.adjusted_survey = adjusted_output

                    st.success("Offsets applied (cumulative)!")
                    st.dataframe(adjusted_output, height=400)

                    # Download adjusted CSV
                    csv_adj = adjusted_output.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Adjusted Survey", data=csv_adj,
                                       file_name="adjusted_survey.csv", mime="text/csv")

                # Undo button


            with col1:

                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=survey_ref.e, y=survey_ref.n, z=survey_ref.tvd,
                    mode='lines', name='Reference Survey', line=dict(color='blue')
                ))
                if st.session_state.adjusted_survey is not None:
                    x_vals = st.session_state.adjusted_survey["East_adj"]
                    y_vals = st.session_state.adjusted_survey["North_adj"]
                    z_vals = st.session_state.adjusted_survey["TVD_adj"]
                else:
                    x_vals = survey_cmp.e
                    y_vals = survey_cmp.n
                    z_vals = survey_cmp.tvd

                fig.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals,
                    mode='lines', name='Comparative Survey (Adjusted)', line=dict(color='red')
                ))

                fig.update_layout(scene=dict(
                    xaxis_title='East [m]',
                    yaxis_title='North [m]',
                    zaxis_title='TVD [m]',
                    zaxis=dict(autorange="reversed"),
                    aspectmode='cube'
                ), height=800)
                st.plotly_chart(fig, use_container_width=True)

        # Displacement vs MD
        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=output['MD'], y=output['Displacement'], mode='lines+markers', name='Displacement'))
            fig2.update_layout(title="3D Displacement vs MD", xaxis_title="MD [m]", yaxis_title="Displacement [m]", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # Delta INC & Delta AZI
        with tab3:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=output['MD'], y=output['Delta_INC'], mode='lines', name='Delta INC'))
            fig3.add_trace(go.Scatter(x=output['MD'], y=output['Delta_AZI'], mode='lines', name='Delta AZI'))
            fig3.update_layout(title="Delta INC & Delta AZI vs MD", xaxis_title="MD [m]", yaxis_title="Degrees", height=400)
            st.plotly_chart(fig3, use_container_width=True)
