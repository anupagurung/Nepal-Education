# --- 0. Import Libraries ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

# --- NEW: Define a professional color palette for consistency ---
COLOR_PALETTE = {
    'primary': '#4A90E2', 'success': '#50E3C2', 'warning': '#F5A623',
    'danger': '#D0021B', 'neutral': '#D3D3D3', 'background': '#F9F9F9',
    'text': '#333333', 'map_bubble': 'rgba(74, 144, 226, 0.7)'
}

# --- 1. Setup & Data Loading ---
print("Step 1: Setting up environment and loading all data...")

# --- GeoJSON Map File Download ---
geojson_filename = "nepal-provinces.geojson"
if not os.path.exists(geojson_filename):
    print(f"Downloading map file: '{geojson_filename}'...")
    url = "https://data.humdata.org/dataset/55f9596c-a33d-4c7f-a6fb-163e2a3423e8/resource/8c85e68b-ed35-48d0-847d-cf47eb4bb6d6/download/geoBoundaries-NPL-ADM1_simplified.geojson"
    try:
        import requests
        response = requests.get(url)
        with open(geojson_filename, 'wb') as f:
            f.write(response.content)
        print("Download successful.")
    except (ImportError, Exception):
        print("Warning: 'requests' library not found. Trying 'curl'. This may fail on standard Windows.")
        os.system(f'curl -L -o {geojson_filename} "{url}"')
    
    if not os.path.exists(geojson_filename):
        print(f"FATAL ERROR: Failed to download map file. Please download it manually from:\n{url}\nand save it as '{geojson_filename}' in the same directory.")
        sys.exit()

# --- Helper Functions ---
def standardize_province_name(name):
    if not isinstance(name, str): return name
    name = name.strip().lower()
    if any(p in name for p in ['province l', 'province1', 'province 1']): return 'Province 1'
    if any(p in name for p in ['province 2', 'province2']): return 'Province 2'
    if any(p in name for p in ['province 3', 'province3']): return 'Province 3'
    if 'gandaki' in name: return 'Gandaki'
    if any(p in name for p in ['province 5', 'province5']): return 'Province 5'
    if 'karnali' in name: return 'Karnali'
    if any(p in name for p in ['province 7', 'province7']): return 'Province 7'
    if 'total' in name or 'nepal' in name: return 'Nepal (Overall)'
    return name

def clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '').str.strip(), errors='coerce')

# --- Load all CSV files ---
file_names = [
    'cleaned_Total Number of Universities and Campuses I Colleges by Province in 2074 BS.csv',
    'cleaned_total-enrollment-at-all-levels_community2011-012-repaired.csv',
    'cleaned_Total Number of Faculties of Higher Education by Province in 2074 BS.csv',
    'cleaned_community-school-student-enrollment-by-province-in-2074-bs.csv',
    'cleaned_districts-wise-student-teacher-ratios-strs-based-on-reported-number-of-teachers-and-students-in-.csv',
    'cleaned_dropout_rates_2011-012.csv',
    'cleaned_educational-institutions-by-province-in-2074-bs.csv',
    'cleaned_Faculty Wise Total Student Enrollment in Higher Education by Province in 207485.csv',
    'cleaned_number-of-persons-with-teaching-license-up-to-fy.-2073-2074.csv',
    'cleaned_percentage-of-teachers-by-training-status-at-institutional-schools-based-on-reporting-of-2009-01.csv',
    'cleaned_total-teachers-in-approved-position-in-community-schools-by-district-in-2074-bs.csv',
    'cleaned_Province -Wise TotalStudent Enrollment by Levelin 2074 BS.csv' 
]
dfs = {}
for f in file_names:
    try:
        key = os.path.basename(f).replace('cleaned_', '').replace('.csv', '')
        dfs[key] = pd.read_csv(f)
    except FileNotFoundError:
        print(f"Warning: Could not find file {f}. It will be skipped.")

# --- 2. Data Processing for "Provincial Overview" Tab ---
print("Step 2: Processing data for 'Provincial Overview' tab...")

# Define keys for easier access
KEY_DROPOUT = 'dropout_rates_2011-012'
KEY_TEACHER_TRAINING = 'percentage-of-teachers-by-training-status-at-institutional-schools-based-on-reporting-of-2009-01'
KEY_STR = 'districts-wise-student-teacher-ratios-strs-based-on-reported-number-of-teachers-and-students-in-'
KEY_FACULTY_ENROLL = 'Faculty Wise Total Student Enrollment in Higher Education by Province in 207485'
KEY_INSTITUTIONS = 'educational-institutions-by-province-in-2074-bs'
KEY_LICENSES = 'number-of-persons-with-teaching-license-up-to-fy.-2073-2074'
KEY_TEACHER_POSTS = 'total-teachers-in-approved-position-in-community-schools-by-district-in-2074-bs'
KEY_UNI_CAMPUSES = 'Total Number of Universities and Campuses I Colleges by Province in 2074 BS'
KEY_GRADE_ENROLL = 'total-enrollment-at-all-levels_community2011-012-repaired'
KEY_SCHOOL_ENROLL = 'community-school-student-enrollment-by-province-in-2074-bs'

provinces_df = pd.DataFrame({'province': ['Province 1', 'Province 2', 'Province 3', 'Gandaki', 'Province 5', 'Karnali', 'Province 7', 'Nepal (Overall)']})

# --- Process base metrics for KPI cards and main charts ---
df_dropout = dfs[KEY_DROPOUT].copy(); df_dropout['province'] = df_dropout['province'].apply(standardize_province_name)
grade_cols = [f'grade {i}' for i in range(1, 11)]; df_dropout['avg_dropout_rate'] = df_dropout[grade_cols].mean(axis=1)
province_dropout = df_dropout.groupby('province')[grade_cols + ['avg_dropout_rate']].mean().reset_index()

df_teacher_training = dfs[KEY_TEACHER_TRAINING].copy(); df_teacher_training['province'] = df_teacher_training['province'].apply(standardize_province_name)
province_teacher_training = df_teacher_training.groupby('province')[['full', 'partial', 'untrained']].mean().reset_index().rename(columns={'untrained': 'untrained_teacher_percent'})

df_str = dfs[KEY_STR].copy(); df_str['province'] = df_str['province'].apply(standardize_province_name); df_str.replace(0, np.nan, inplace=True)
province_str = df_str.groupby('province')[['total']].mean().reset_index().rename(columns={'total': 'avg_str_reported'})

df_higher_ed = dfs[KEY_FACULTY_ENROLL].copy(); df_higher_ed['province'] = df_higher_ed['description'].apply(standardize_province_name)
df_higher_ed['total_higher_ed_students'] = clean_numeric(df_higher_ed['total'])
total_he_students = df_higher_ed.loc[df_higher_ed['province'] == 'Nepal (Overall)', 'total_higher_ed_students'].iloc[0]
df_higher_ed['higher_ed_share_percent'] = (df_higher_ed['total_higher_ed_students'] / total_he_students) * 100

df_institutions = dfs[KEY_INSTITUTIONS].copy(); df_institutions.columns = [col.lower().strip() for col in df_institutions.columns]; df_institutions.rename(columns={'province1': 'Province 1', 'province 2': 'Province 2', 'province 3': 'Province 3', 'gandaki': 'Gandaki', 'province 5': 'Province 5', 'karnali': 'Karnali', 'province 7': 'Province 7', 'total': 'Nepal (Overall)'}, inplace=True)
institutions_T = df_institutions.set_index('description').T.reset_index().rename(columns={'index': 'province'}); institutions_T['province'] = institutions_T['province'].apply(standardize_province_name)
institutions_T.rename(columns=lambda x: x.strip().lower().replace(' ', '_').replace('/', '_'), inplace=True); institutions_T['campuses'] = clean_numeric(institutions_T['campuses'])

df_licenses = dfs[KEY_LICENSES].copy(); df_licenses['province'] = df_licenses['province'].apply(standardize_province_name); df_licenses['total'] = clean_numeric(df_licenses['total'])

df_teachers = dfs[KEY_TEACHER_POSTS].copy(); df_teachers['province'] = df_teachers['province no'].apply(standardize_province_name)
teacher_cols = [col for col in df_teachers.columns if 'teacher' in col]; province_teachers = df_teachers.groupby('province')[teacher_cols].sum().reset_index()
province_teachers['total_approved_posts'] = province_teachers[['approved teacher posts primary', 'approved teacher posts lower secondary', 'approved teacher posts secondary']].sum(axis=1)
province_teachers['total_rahat_posts'] = province_teachers[['rahat teacher posts primary', 'rahat teacher posts lower secondary', 'rahat teacher posts secondary']].sum(axis=1)

df_school_enroll = dfs[KEY_SCHOOL_ENROLL].copy()
df_school_enroll['province'] = df_school_enroll['province'].apply(standardize_province_name)
enrollment_cols = ['basic (1-5) total', 'basic (6-8) total', 'secondary (9-10) total', 'secondary (11-12) total']
for col in enrollment_cols:
    df_school_enroll[col] = clean_numeric(df_school_enroll[col])
df_school_enroll['total_school_students'] = df_school_enroll[enrollment_cols].sum(axis=1)
province_school_enroll = df_school_enroll[['province', 'total_school_students']]

# --- Create the master dataframe by merging all processed sources ---
master_df = provinces_df.merge(
    province_dropout[['province', 'avg_dropout_rate']], on='province', how='left'
).merge(
    province_teacher_training, on='province', how='left'
).merge(
    province_str, on='province', how='left'
).merge(
    df_higher_ed[['province', 'total_higher_ed_students', 'higher_ed_share_percent']], on='province', how='left'
).merge(
    institutions_T[['province', 'campuses']], on='province', how='left'
).merge(
    df_licenses[['province', 'total']].rename(columns={'total': 'total_licenses'}), on='province', how='left'
).merge(
    province_teachers[['province', 'total_approved_posts', 'total_rahat_posts']], on='province', how='left'
).merge(
    province_school_enroll, on='province', how='left'
)

master_df.set_index('province', inplace=True)

master_df['total_all_students'] = master_df['total_school_students'].fillna(0) + master_df['total_higher_ed_students'].fillna(0)

# --- Process faculty enrollment data ---
df_faculty_enroll = dfs[KEY_FACULTY_ENROLL].copy(); df_faculty_enroll['province'] = df_faculty_enroll['description'].apply(standardize_province_name)
faculty_cols = ['agriculture', 'education', 'engineering', 'forestry', 'hss', 'law', 'management', 'medicine', 'science_technology', 'sanskrit']
for col in faculty_cols: df_faculty_enroll[col] = clean_numeric(df_faculty_enroll[col])
faculty_enroll_long = df_faculty_enroll.melt(id_vars='province', value_vars=faculty_cols, var_name='faculty', value_name='students')

# --- Process University Campus Data ---
df_uni = dfs[KEY_UNI_CAMPUSES].copy()
df_uni = df_uni[~df_uni['university'].str.contains("Total", na=False)].copy()
uni_name_map = {'Tr bhuvan University': 'Tribhuvan University', 'NepalSanskr t University': 'Nepal Sanskrit University', 'PurbanchalUniversity': 'Purbanchal University', 'LumbiniBauddha University': 'Lumbini Bauddha University', 'KarnaliAcademy HealthSciences': 'Karnali Academy Health Sciences'}
df_uni['university'] = df_uni['university'].str.strip().replace(uni_name_map)
df_uni.rename(columns={'province 1': 'Province 1', 'province 2': 'Province 2', 'province 3': 'Province 3', 'gandaki': 'Gandaki', 'province 5': 'Province 5', 'karnali': 'Karnali', 'province 7': 'Province 7'}, inplace=True)
province_cols_for_melt = ['Province 1', 'Province 2', 'Province 3', 'Gandaki', 'Province 5', 'Karnali', 'Province 7']
df_uni_long = df_uni.melt(id_vars=['university'], value_vars=province_cols_for_melt, var_name='province', value_name='campuses')
df_uni_long['campuses'] = clean_numeric(df_uni_long['campuses']).fillna(0).astype(int)

# --- Process Grade-wise Enrollment Data ---
df_grade_enroll = dfs[KEY_GRADE_ENROLL].copy()
df_grade_enroll = df_grade_enroll[df_grade_enroll['province'] != 'Unknown'].copy()
df_grade_enroll['province'] = df_grade_enroll['province'].apply(standardize_province_name)
actual_grade_cols = ['grade 1', 'grade 2', 'grade 3', 'grade 4', 'grade 5', 'grade 6', 'grade 7', 'grade 8', 'grade 9', 'grade 10', 'grade11', 'grade 12']
for col in actual_grade_cols: df_grade_enroll[col] = clean_numeric(df_grade_enroll[col])
df_grade_enroll_by_province = df_grade_enroll.groupby('province')[actual_grade_cols].sum()
df_grade_enroll_by_province.loc['Nepal (Overall)'] = df_grade_enroll_by_province.sum()


# --- 3. Data Processing for "Enrollment Map" Tab ---
print("Step 3: Processing data for 'Enrollment Map' tab...")

df_enrollment_raw = dfs['Province -Wise TotalStudent Enrollment by Levelin 2074 BS'].copy()
df_enrollment_raw.columns = [standardize_province_name(col) for col in df_enrollment_raw.columns]
df_enrollment_raw.rename(columns={'educational level': 'level'}, inplace=True)
map_data_df = df_enrollment_raw.melt(id_vars='level', var_name='province', value_name='enrollment')
map_data_df['enrollment'] = map_data_df['enrollment'].astype(str).str.replace(r'[,./]', '', regex=True).pipe(pd.to_numeric, errors='coerce').fillna(0).astype(int)
levels_to_include = ['Basic(1-5)', 'Basic(6-8)', 'Secondary (9-10)', 'Secondary (11-12)']
map_data_df = map_data_df[map_data_df['level'].isin(levels_to_include)].copy()
map_data_df['level'] = map_data_df['level'].str.replace('(', ' (', regex=False)

province_name_map = {'Province 1': 'Koshi', 'Province 2': 'Madhesh', 'Province 3': 'Bagmati', 'Gandaki': 'Gandaki', 'Province 5': 'Lumbini', 'Karnali': 'Karnali', 'Province 7': 'Sudurpashchim'}
province_centroids = {'Province 1': {'lat': 27.1, 'lon': 87.3}, 'Province 2': {'lat': 26.9, 'lon': 85.9}, 'Province 3': {'lat': 27.8, 'lon': 85.4}, 'Gandaki': {'lat': 28.5, 'lon': 84.0}, 'Province 5': {'lat': 28.0, 'lon': 82.8}, 'Karnali': {'lat': 29.3, 'lon': 82.2}, 'Province 7': {'lat': 29.3, 'lon': 81.0}}
map_data_df['geojson_province_name'] = map_data_df['province'].map(province_name_map)
map_data_df['lat'] = map_data_df['province'].apply(lambda p: province_centroids.get(p, {}).get('lat'))
map_data_df['lon'] = map_data_df['province'].apply(lambda p: province_centroids.get(p, {}).get('lon'))

# --- 4. Generate Static Map Figure ---
print("Step 4: Generating static map figure...")

with open(geojson_filename) as f:
    nepal_geojson = json.load(f)

fig_map_enrollment = go.Figure()
fig_map_enrollment.add_trace(go.Choroplethmapbox(
    geojson=nepal_geojson, locations=map_data_df['geojson_province_name'].unique(), featureidkey="properties.shapeName",
    z=[1]*len(map_data_df['geojson_province_name'].unique()), colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
    showscale=False, marker_opacity=0.3, hoverinfo='skip'
))

provinces_on_map = sorted([p for p in map_data_df['province'].unique() if p != 'Nepal (Overall)'])
province_color_sequence = px.colors.qualitative.Vivid 
province_color_map = {province: province_color_sequence[i % len(province_color_sequence)] for i, province in enumerate(provinces_on_map)}
levels = map_data_df['level'].unique()

for i, level in enumerate(levels):
    df_level = map_data_df[map_data_df['level'] == level].copy()
    bubble_colors = df_level['province'].map(province_color_map)
    fig_map_enrollment.add_trace(go.Scattermapbox(
        lat=df_level['lat'], lon=df_level['lon'], mode='markers+text',
        marker=go.scattermapbox.Marker(size=np.sqrt(df_level['enrollment']) / 10, color=bubble_colors, sizemode='diameter'),
        text=df_level['province'], textposition='top center', textfont=dict(color=COLOR_PALETTE['text'], size=10),
        hovertemplate='<b>%{text}</b><br>Enrollment: %{customdata:,.0f}<extra></extra>',
        customdata=df_level['enrollment'], name=level, visible=(i == 0)
    ))

buttons = [{'label': level, 'method': "restyle", 'args': [{"visible": [True] + [level == l for l in levels]}]} for i, level in enumerate(levels)]
fig_map_enrollment.update_layout(
    title_text='<b>Student Enrollment by Educational Level (2074 BS)</b><br><sup>Use dropdown to select level | Bubble Size = Enrollment</sup>',
    mapbox_style="carto-positron", mapbox_zoom=5.5, mapbox_center={"lat": 28.3949, "lon": 84.1240},
    margin={"r":0, "t":50, "l":0, "b":0}, showlegend=False,
    updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=0.99, yanchor="top")]
)

print("All data processing and static figure generation complete.")

# --- 5. Initialize Dash App & Define Layout ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Nepal Education Dashboard", style={'color': COLOR_PALETTE['text']})), className="my-4"),
    dbc.Row(dbc.Col([
        html.Label("Select a Province to Analyze:", className="fw-bold", style={'color': COLOR_PALETTE['text']}),
        dcc.Dropdown(id='province-dropdown', options=[{'label': i, 'value': i} for i in master_df.index], value='Nepal (Overall)', clearable=False)
    ])),
    
    dbc.Tabs([
        dbc.Tab(label='Provincial Overview', children=[
            dbc.Row(id='kpi-cards', className="my-4"),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='faculty-enrollment-chart')), width=12, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='higher-ed-donut')), width=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(id='teacher-training-chart')), width=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='str-chart')), width=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(id='teacher-posts-chart')), width=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='dropout-rate-chart')), width=6, className="mb-4"),
                dbc.Col(dbc.Card(dcc.Graph(id='university-campus-chart')), width=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(id='grade-enrollment-chart')), width=12, className="mb-4"),
            ])
        ]),
        dbc.Tab(label='Enrollment Map', children=[
            dbc.Row(dbc.Col(dcc.Graph(id='enrollment-map', figure=fig_map_enrollment, style={'height': '80vh'}), className="mt-4"))
        ])
    ])
], fluid=True, className="dbc", style={'backgroundColor': COLOR_PALETTE['background']})

# --- 6. Helper function for KPI cards ---
def create_kpi_card(title, value, color_hex):
    display_value = f"{value:,.1f}" if isinstance(value, float) else f"{value:,.0f}" if pd.notna(value) else "N/A"
    return dbc.Col(dbc.Card(dbc.CardBody([html.P(title, className="card-title text-muted"), html.H4(display_value, className="card-text", style={'color': color_hex})])))

# --- 7. Callback for "Provincial Overview" Tab ---
@app.callback(
    [Output('kpi-cards', 'children'), Output('higher-ed-donut', 'figure'), Output('teacher-training-chart', 'figure'),
     Output('str-chart', 'figure'), Output('dropout-rate-chart', 'figure'),
     Output('faculty-enrollment-chart', 'figure'), Output('teacher-posts-chart', 'figure'),
     Output('university-campus-chart', 'figure'), Output('grade-enrollment-chart', 'figure')],
    [Input('province-dropdown', 'value')]
)
def update_overview_tab(selected_province):
    province_data = master_df.loc[selected_province]
    kpi_cards = dbc.Row([
        create_kpi_card("Total Students (All Levels)", province_data['total_all_students'], COLOR_PALETTE['primary']),
        create_kpi_card("Avg. Dropout Rate (%)", province_data['avg_dropout_rate'], COLOR_PALETTE['danger']),
        create_kpi_card("Higher Ed Campuses", province_data['campuses'], COLOR_PALETTE['success']),
        create_kpi_card("Untrained Teachers (%)", province_data['untrained_teacher_percent'], COLOR_PALETTE['warning']),
    ])

    # Donut Chart
    donut_colors = [COLOR_PALETTE['neutral']] * len(master_df.index)
    if selected_province != 'Nepal (Overall)':
        donut_colors[master_df.index.get_loc(selected_province)] = COLOR_PALETTE['primary']
    fig_donut = go.Figure(data=[go.Pie(labels=master_df.index, values=master_df['higher_ed_share_percent'], hole=.4, marker_colors=donut_colors, texttemplate='%{label}<br>%{percent:.1%}', hovertemplate='<b>%{label}</b><br>Share: %{percent:.1%}<br>Students: %{value:,.0f}<extra></extra>')])
    fig_donut.update_layout(title_text=f'<b>Higher Ed Student Share: {selected_province}</b>', showlegend=False, uniformtext_minsize=10, paper_bgcolor='white', plot_bgcolor='white')
    
    # === MODIFICATION START ===
    
    # Teacher Training Chart
    tt_data = master_df[['full', 'partial', 'untrained_teacher_percent']].drop('Nepal (Overall)')
    # If a specific province is selected, filter the data
    if selected_province != 'Nepal (Overall)':
        tt_data = tt_data.loc[[selected_province]]
        chart_title = f'<b>Teacher Training Status in {selected_province}</b>'
    else:
        chart_title = '<b>Teacher Training Status by Province</b>'
    
    fig_teacher_training = px.bar(tt_data, x=tt_data.index, y=['full', 'partial', 'untrained_teacher_percent'], 
                                  title=chart_title, 
                                  labels={'value': 'Percentage of Teachers (%)', 'province': 'Province'}, 
                                  color_discrete_map={'full': COLOR_PALETTE['success'], 'partial': COLOR_PALETTE['warning'], 'untrained_teacher_percent': COLOR_PALETTE['danger']})
    fig_teacher_training.update_layout(legend_title_text='Training Status', paper_bgcolor='white', plot_bgcolor='white')
    
    # STR Chart
    str_data = master_df['avg_str_reported'].drop('Nepal (Overall)').sort_values()
    # If a specific province is selected, filter the data
    if selected_province != 'Nepal (Overall)':
        str_data = str_data.loc[[selected_province]]
        chart_title_str = f'<b>Average Student-Teacher Ratio in {selected_province}</b>'
    else:
        chart_title_str = '<b>Average Student-Teacher Ratio</b>'

    fig_str = px.bar(str_data, x=str_data.index, y=str_data.values, 
                     title=chart_title_str, 
                     labels={'y': 'Students per Teacher', 'x': 'Province'}, 
                     text_auto='.1f',
                     color=str_data.index,
                     color_discrete_sequence=px.colors.qualitative.Vivid)
    fig_str.update_layout(paper_bgcolor='white', plot_bgcolor='white', showlegend=False)

    # === MODIFICATION END ===
    
    # Dropout Rate Chart
    fig_dropout = go.Figure().update_layout(title_text='Select a Province to see Dropout Rates by Grade', paper_bgcolor='white', plot_bgcolor='white', xaxis={'visible': False}, yaxis={'visible': False})
    if selected_province in province_dropout['province'].values:
        dropout_data = province_dropout.set_index('province').loc[selected_province][grade_cols]
        fig_dropout = px.bar(dropout_data, x=dropout_data.index, y=dropout_data.values, 
                             title=f'<b>Dropout Rates (%) by Grade in {selected_province}</b>', 
                             labels={'y': 'Dropout Rate (%)', 'x': 'Grade Level'}, 
                             text_auto='.1f',
                             color=dropout_data.index,
                             color_discrete_sequence=px.colors.sequential.Plasma)
        fig_dropout.update_layout(paper_bgcolor='white', plot_bgcolor='white', showlegend=False)
        
    # Faculty Enrollment Chart
    faculty_data = faculty_enroll_long[faculty_enroll_long['province'] == selected_province].sort_values('students', ascending=False)
    fig_faculty = px.bar(faculty_data, x='faculty', y='students', 
                         title=f'<b>Higher Ed Enrollment by Faculty in {selected_province}</b>', 
                         labels={'students': 'Number of Students', 'faculty': 'Faculty'}, 
                         text_auto=True,
                         color='faculty',
                         color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_faculty.update_layout(xaxis_tickangle=-45, paper_bgcolor='white', plot_bgcolor='white', showlegend=False)
    
    # Teacher Posts Chart
    fig_posts = go.Figure().update_layout(title_text='Select a Province to see Teacher Posts', paper_bgcolor='white', plot_bgcolor='white', xaxis={'visible': False}, yaxis={'visible': False})
    if selected_province in province_teachers['province'].values:
        teacher_posts_data = province_teachers.set_index('province').loc[selected_province][['total_approved_posts', 'total_rahat_posts']]
        fig_posts = go.Figure(data=[go.Bar(x=['Approved Posts', 'Rahat Posts'], y=teacher_posts_data.values, text=[f'{v:,.0f}' for v in teacher_posts_data.values], textposition='auto', marker_color=[COLOR_PALETTE['success'], COLOR_PALETTE['warning']])])
        fig_posts.update_layout(title=f'<b>Teacher Posts (Community) in {selected_province}</b>', xaxis_title='Post Type', yaxis_title='Number of Posts', paper_bgcolor='white', plot_bgcolor='white')
        
    # University Campus Chart
    uni_data = df_uni_long[df_uni_long['province'] == selected_province]
    uni_data = uni_data[uni_data['campuses'] > 0].sort_values('campuses', ascending=False)
    fig_uni = px.bar(uni_data, x='university', y='campuses', 
                     text_auto=True, 
                     color='university',
                     color_discrete_sequence=px.colors.qualitative.Antique,
                     title=f"<b>University Campuses in {selected_province}</b>",
                     labels={'campuses': 'Number of Campuses/Colleges', 'university': 'University'})
    fig_uni.update_layout(xaxis_tickangle=-45, paper_bgcolor='white', plot_bgcolor='white', showlegend=False)

    # Grade-wise Enrollment Chart
    grade_enroll_data = df_grade_enroll_by_province.loc[selected_province].reset_index()
    grade_enroll_data.columns = ['grade', 'students']
    grade_enroll_data['grade_num'] = grade_enroll_data['grade'].str.extract(r'(\d+)').astype(int)
    grade_enroll_data = grade_enroll_data.sort_values('grade_num')
    grade_enroll_data['grade'] = grade_enroll_data['grade'].str.replace('grade', 'Grade ').str.replace('11', ' 11').str.replace('12', ' 12').str.strip()
    fig_grade_enroll = px.bar(grade_enroll_data, x='grade', y='students', 
                              text_auto=True, 
                              color='grade',
                              color_discrete_sequence=px.colors.sequential.Viridis,
                              title=f"<b>Enrollment by Grade in {selected_province} (2011/12)</b>",
                              labels={'students': 'Total Enrolled Students', 'grade': 'Grade Level'})
    fig_grade_enroll.update_traces(texttemplate='%{y:,.0f}')
    fig_grade_enroll.update_layout(paper_bgcolor='white', plot_bgcolor='white', showlegend=False)
        
    return kpi_cards, fig_donut, fig_teacher_training, fig_str, fig_dropout, fig_faculty, fig_posts, fig_uni, fig_grade_enroll

# --- 8. Run the App ---
if __name__ == '__main__':
    print("\nStep 5: Starting Dash server...")
    print("Go to http://127.0.0.1:8050 to view the dashboard.")
    app.run(debug=True, port=8050)