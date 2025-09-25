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

            # Offsets must be defined first
            with col2:
                st.markdown("### Adjust Comparative Survey with Offsets")

                x_offset = st.number_input("Adjust Easting (X)", value=0.0, step=1.0, key="x_offset")
                y_offset = st.number_input("Adjust Northing (Y)", value=0.0, step=1.0, key="y_offset")
                z_offset = st.number_input("Adjust TVD (Z)", value=0.0, step=1.0, key="z_offset")

                adjusted_output = None
                if st.button("Apply Offsets"):
                    # Apply offsets
                    e_adj = survey_cmp.e + x_offset
                    n_adj = survey_cmp.n + y_offset
                    tvd_adj = survey_cmp.tvd + z_offset

                    # Calculate step deltas
                    dN = np.gradient(n_adj)
                    dE = np.gradient(e_adj)
                    dTVD = np.gradient(tvd_adj)

                    # Compute INC and AZI
                    inc_adj = np.degrees(np.arctan2(np.sqrt(dN ** 2 + dE ** 2), dTVD))
                    azi_adj = (np.degrees(np.arctan2(dE, dN)) + 360) % 360

                    adjusted_output = pd.DataFrame({
                        "MD": survey_cmp.md,
                        "North_adj": n_adj,
                        "East_adj": e_adj,
                        "TVD_adj": tvd_adj,
                        "INC_adj": inc_adj,
                        "AZI_adj": azi_adj
                    })

                    st.success("Offsets applied and MD/INC/AZI recalculated!")
                    st.dataframe(adjusted_output, height=400)

                    # Download adjusted CSV
                    csv_adj = adjusted_output.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Adjusted Survey", data=csv_adj,
                                       file_name="adjusted_survey.csv", mime="text/csv")

            with col1:
                # Plot (uses offsets whether applied or not)
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=survey_ref.e, y=survey_ref.n, z=survey_ref.tvd,
                    mode='lines', name='Reference Survey', line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter3d(
                    x=survey_cmp.e + x_offset,
                    y=survey_cmp.n + y_offset,
                    z=survey_cmp.tvd + z_offset,
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
