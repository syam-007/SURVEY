import pandas as pd
import welleng as we
import plotly.graph_objects as go
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# -----------------------------
# Step 1: Ask user to select Excel file
# -----------------------------
Tk().withdraw()  # hide the root Tk window
file_path = askopenfilename(
    title="Select Excel File",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

if not file_path:
    raise SystemExit("No file selected. Exiting.")

# -----------------------------
# Step 2: Read Excel file
# -----------------------------
df = pd.read_excel(file_path)

# Ensure columns exist
required_columns = ["md", "inc", "azi"]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Excel file must contain columns: {required_columns}")

# -----------------------------
# Step 3: Sort MD and remove duplicates
# -----------------------------
df = df.sort_values("md").drop_duplicates(subset="md", keep="first")

# Check for strictly increasing MD
if not all(df["md"].diff().dropna() > 0):
    raise ValueError("MD values must be strictly increasing for interpolation")

# -----------------------------
# Step 4: Create WellEng survey
# -----------------------------
s = we.survey.Survey(
    md=df["md"].tolist(),
    inc=df["inc"].tolist(),
    azi=df["azi"].tolist()
)

# -----------------------------
# Step 5: Interpolate survey
# -----------------------------
s_interp = s.interpolate_survey(step=5)  # step in meters

# -----------------------------
# Step 6: Plot survey
# -----------------------------
fig = go.Figure()

# Interpolated survey line
fig.add_trace(
    go.Scatter3d(
        x=s_interp.x,
        y=s_interp.y,
        z=s_interp.z,
        mode='lines',
        line=dict(color='blue', width=3),
        name='Survey Interpolated'
    )
)

# Original survey points
fig.add_trace(
    go.Scatter3d(
        x=s.x,
        y=s.y,
        z=s.z,
        mode='markers',
        marker=dict(color='red', size=3),
        name='Original Survey'
    )
)

# Reverse Z axis to match well orientation
fig.update_scenes(zaxis_autorange="reversed")

fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (MD)',
    ),
    title="Well Survey Plot",
    margin=dict(l=0, r=0, b=0, t=30)
)

fig.show()
