# -*- coding: utf-8 -*-
"""
EduReform Nepal - Interactive Dashboard with Streamlit (All Provinces Comparative View) - FINAL

This single script creates a comprehensive, multi-tabbed dashboard for analyzing Nepal's education data using Streamlit.
- Tab 1: "Provincial Overview" now provides consolidated, comparative charts for all provinces.
- Tab 2: "Enrollment Map" provides an interactive map to visualize student enrollment.
"""

# --- 0. Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import requests

# --- NEW: Define a professional color palette for consistency ---
COLOR_PALETTE = {
    'primary': '#4A90E2', 'success': '#50E3C2', 'warning': '#F5A623',
    'danger': '#D0021B', 'neutral': '#D3D3D3', 'background': '#F9F9F9',
    'text': '#333333'
}

# --- 1. Setup & Data Loading (Cached for performance) ---
@st.cache_data
def load_all_data():
    """
    Loads, cleans, and processes all data required for the dashboard.
    This function is cached to run only once.
    """
    # --- GeoJSON Map File Download ---
    geojson_filename = "nepal-provinces.geojson"
    if not os.path.exists(geojson_filename):
        with st.spinner(f"Downloading map file: '{geojson_filename}'..."):
            url = "https://data.humdata.org/dataset/55f9596c-a33d-4c7f-a6fb-163e2a3423e8/resource/8c85e68b-ed35-48d0-847d-cf47eb4bb6d6/download/geoBoundaries-NPL-ADM1_simplified.geojson"
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(geojson_filename, 'wb') as f:
                    f.write(response.content)
            except requests.exceptions.RequestException as e:
                st.error(f"FATAL ERROR: Failed to download map file. Please check your internet connection or download it manually from:\n{url}\nand save it as '{geojson_filename}'.")
                st.stop()

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
        'cleaned_Total Number of Faculties of Higher Education by Province in 2074 BS.csv',
        'cleaned_community-school-student-enrollment-by-province-in-2074-bs.csv',
        'cleaned_districts-wise-student-teacher-ratios-strs-based-on-reported-number-of-teachers-and-students-in-.csv',
        'cleaned_dropout_rates_2011-012.csv',
        'cleaned_educational-institutions-by-province-in-2074-bs.csv',
        'cleaned_gross-enrollment-rate-ger-by-province-in-2074-bs.csv',
        'cleaned_Faculty Wise Total Student Enrollment in Higher Education by Province in 207485.csv',
        'cleaned_number-of-persons-with-teaching-license-up-to-fy.-2073-2074.csv',
        'cleaned_percentage-of-teachers-by-training-status-at-institutional-schools-based-on-reporting-of-2009-01.csv',
        'cleaned_total-teachers-in-approved-position-in-community-schools-by-district-in-2074-bs.csv',
        'cleaned_Province -Wise TotalStudent Enrollment by Levelin 2074 BS.csv'
    ]
    dfs = {}
    for f in file_names:
        try:
            key = os.path.basename(f).replace('cleaned_', '').split('.')[0][:20]
            dfs[key] = pd.read_csv(f)
        except FileNotFoundError:
            st.error(f"Error: Could not find required file '{f}'. Please ensure all CSV data files are in the same directory as the script.")
            st.stop()
    
    # --- Data Processing ---
    KEY_DROPOUT = 'dropout_rates_2011-0'; KEY_TEACHER_TRAINING = 'percentage-of-teache'; KEY_STR = 'districts-wise-stude'; KEY_GER = 'gross-enrollment-rat'; KEY_FACULTY_ENROLL = 'Faculty Wise Total S'; KEY_INSTITUTIONS = 'educational-institut'; KEY_LICENSES = 'number-of-persons-wi'; KEY_TEACHER_POSTS = 'total-teachers-in-ap'
    provinces_df = pd.DataFrame({'province': ['Province 1', 'Province 2', 'Province 3', 'Gandaki', 'Province 5', 'Karnali', 'Province 7', 'Nepal (Overall)']})
    df_dropout = dfs[KEY_DROPOUT].copy(); df_dropout['province'] = df_dropout['province'].apply(standardize_province_name); grade_cols = [f'grade {i}' for i in range(1, 11)]; df_dropout['avg_dropout_rate'] = df_dropout[grade_cols].mean(axis=1); province_dropout = df_dropout.groupby('province')[grade_cols + ['avg_dropout_rate']].mean().reset_index()
    df_teacher_training = dfs[KEY_TEACHER_TRAINING].copy(); df_teacher_training['province'] = df_teacher_training['province'].apply(standardize_province_name); province_teacher_training = df_teacher_training.groupby('province')[['full', 'partial', 'untrained']].mean().reset_index().rename(columns={'untrained': 'untrained_teacher_percent'})
    df_str = dfs[KEY_STR].copy(); df_str['province'] = df_str['province'].apply(standardize_province_name); df_str.replace(0, np.nan, inplace=True); province_str = df_str.groupby('province')[['total']].mean().reset_index().rename(columns={'total': 'avg_str_reported'})
    df_ger = dfs[KEY_GER].copy(); df_ger['province'] = df_ger['province'].apply(standardize_province_name); ger_secondary = df_ger[['province', 'secondary(9-10) girls', 'secondary(9-10) boys']].rename(columns={'secondary(9-10) girls': 'ger_sec_girls', 'secondary(9-10) boys': 'ger_sec_boys'})
    df_higher_ed = dfs[KEY_FACULTY_ENROLL].copy(); df_higher_ed['province'] = df_higher_ed['description'].apply(standardize_province_name); df_higher_ed['total_higher_ed_students'] = clean_numeric(df_higher_ed['total']); total_he_students = df_higher_ed.loc[df_higher_ed['province'] == 'Nepal (Overall)', 'total_higher_ed_students'].iloc[0]; df_higher_ed['higher_ed_share_percent'] = (df_higher_ed['total_higher_ed_students'] / total_he_students) * 100
    df_institutions = dfs[KEY_INSTITUTIONS].copy(); df_institutions.columns = [col.lower().strip() for col in df_institutions.columns]; df_institutions.rename(columns={'province1': 'Province 1', 'province 2': 'Province 2', 'province 3': 'Province 3', 'gandaki': 'Gandaki', 'province 5': 'Province 5', 'karnali': 'Karnali', 'province 7': 'Province 7', 'total': 'Nepal (Overall)'}, inplace=True); institutions_T = df_institutions.set_index('description').T.reset_index().rename(columns={'index': 'province'}); institutions_T['province'] = institutions_T['province'].apply(standardize_province_name); institutions_T.rename(columns=lambda x: x.strip().lower().replace(' ', '_').replace('/', '_'), inplace=True); institutions_T['campuses'] = clean_numeric(institutions_T['campuses'])
    df_licenses = dfs[KEY_LICENSES].copy(); df_licenses['province'] = df_licenses['province'].apply(standardize_province_name); df_licenses['total'] = clean_numeric(df_licenses['total'])
    df_teachers = dfs[KEY_TEACHER_POSTS].copy(); df_teachers['province'] = df_teachers['province no'].apply(standardize_province_name); teacher_cols = [col for col in df_teachers.columns if 'teacher' in col]; province_teachers = df_teachers.groupby('province')[teacher_cols].sum().reset_index(); province_teachers['total_approved_posts'] = province_teachers[['approved teacher posts primary', 'approved teacher posts lower secondary', 'approved teacher posts secondary']].sum(axis=1); province_teachers['total_rahat_posts'] = province_teachers[['rahat teacher posts primary', 'rahat teacher posts lower secondary', 'rahat teacher posts secondary']].sum(axis=1)
    master_df = provinces_df.merge(province_dropout[['province', 'avg_dropout_rate']], on='province', how='left').merge(province_teacher_training, on='province', how='left').merge(province_str, on='province', how='left').merge(ger_secondary, on='province', how='left').merge(df_higher_ed[['province', 'total_higher_ed_students', 'higher_ed_share_percent']], on='province', how='left').merge(institutions_T[['province', 'campuses']], on='province', how='left').merge(df_licenses[['province', 'total']].rename(columns={'total': 'total_licenses'}), on='province', how='left').merge(province_teachers[['province', 'total_approved_posts', 'total_rahat_posts']], on='province', how='left').set_index('province')
    
    df_faculty_enroll = dfs[KEY_FACULTY_ENROLL].copy()
    df_faculty_enroll['province'] = df_faculty_enroll['description'].apply(standardize_province_name)
    faculty_cols = ['agriculture', 'education', 'engineering', 'forestry', 'hss', 'law', 'management', 'medicine', 'science_technology', 'sanskrit']
    for col in faculty_cols:
        df_faculty_enroll[col] = clean_numeric(df_faculty_enroll[col])
    faculty_enroll_long = df_faculty_enroll.melt(id_vars='province', value_vars=faculty_cols, var_name='faculty', value_name='students')
    
    # Data for Enrollment Map
    df_enrollment_raw = dfs['Province -Wise Total'].copy(); df_enrollment_raw.columns = [standardize_province_name(col) for col in df_enrollment_raw.columns]; df_enrollment_raw.rename(columns={'educational level': 'level'}, inplace=True); map_data_df = df_enrollment_raw.melt(id_vars='level', var_name='province', value_name='enrollment'); map_data_df['enrollment'] = map_data_df['enrollment'].astype(str).str.replace(r'[,./]', '', regex=True).pipe(pd.to_numeric, errors='coerce').fillna(0).astype(int); levels_to_include = ['Basic(1-5)', 'Basic(6-8)', 'Secondary (9-10)', 'Secondary (11-12)']; map_data_df = map_data_df[map_data_df['level'].isin(levels_to_include)].copy(); map_data_df['level'] = map_data_df['level'].str.replace('(', ' (', regex=False); province_name_map = {'Province 1': 'Koshi', 'Province 2': 'Madhesh', 'Province 3': 'Bagmati', 'Gandaki': 'Gandaki', 'Province 5': 'Lumbini', 'Karnali': 'Karnali', 'Province 7': 'Sudurpashchim'}; province_centroids = {'Province 1': {'lat': 27.1, 'lon': 87.3}, 'Province 2': {'lat': 26.9, 'lon': 85.9}, 'Province 3': {'lat': 27.8, 'lon': 85.4}, 'Gandaki': {'lat': 28.5, 'lon': 84.0}, 'Province 5': {'lat': 28.0, 'lon': 82.8}, 'Karnali': {'lat': 29.3, 'lon': 82.2}, 'Province 7': {'lat': 29.3, 'lon': 81.0}}; map_data_df['geojson_province_name'] = map_data_df['province'].map(province_name_map); map_data_df['lat'] = map_data_df['province'].apply(lambda p: province_centroids.get(p, {}).get('lat')); map_data_df['lon'] = map_data_df['province'].apply(lambda p: province_centroids.get(p, {}).get('lon'))

    # Prepare data for teacher posts chart
    teacher_posts_long = province_teachers.melt(id_vars='province', value_vars=['total_approved_posts', 'total_rahat_posts'], var_name='post_type', value_name='count')
    teacher_posts_long['post_type'] = teacher_posts_long['post_type'].replace({'total_approved_posts': 'Approved Posts', 'total_rahat_posts': 'Rahat Posts'})
    
    return master_df, faculty_enroll_long, province_dropout, teacher_posts_long, map_data_df, geojson_filename, grade_cols


# --- 2. Generate Figures ---
def generate_enrollment_map(map_data_df, geojson_path):
    # This function is unchanged and works as before
    with open(geojson_path) as f:
        nepal_geojson = json.load(f)
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(geojson=nepal_geojson, locations=map_data_df['geojson_province_name'].unique(), featureidkey="properties.shapeName", z=[1]*len(map_data_df['geojson_province_name'].unique()), colorscale=[[0, 'lightgrey'], [1, 'lightgrey']], showscale=False, marker_opacity=0.3, hoverinfo='skip'))
    provinces_on_map = sorted([p for p in map_data_df['province'].unique() if p != 'Nepal (Overall)'])
    province_color_sequence = px.colors.qualitative.Vivid
    province_color_map = {province: province_color_sequence[i % len(province_color_sequence)] for i, province in enumerate(provinces_on_map)}
    levels = map_data_df['level'].unique()
    for i, level in enumerate(levels):
        df_level = map_data_df[map_data_df['level'] == level].copy()
        bubble_colors = df_level['province'].map(province_color_map)
        fig.add_trace(go.Scattermapbox(lat=df_level['lat'], lon=df_level['lon'], mode='markers+text', marker=go.scattermapbox.Marker(size=np.sqrt(df_level['enrollment']) / 10, color=bubble_colors, sizemode='diameter'), text=df_level['province'], textposition='top center', textfont=dict(color=COLOR_PALETTE['text'], size=10), hovertemplate='<b>%{text}</b><br>Enrollment: %{customdata:,.0f}<extra></extra>', customdata=df_level['enrollment'], name=level, visible=(i == 0)))
    buttons = [{'label': level, 'method': "restyle", 'args': [{"visible": [True] + [level == l for l in levels]}]} for i, level in enumerate(levels)]
    fig.update_layout(title_text='<b>Student Enrollment by Educational Level</b><br><sup>Use dropdown to select level | Bubble Size = Enrollment</sup>', mapbox_style="carto-positron", mapbox_zoom=5.5, mapbox_center={"lat": 28.3949, "lon": 84.1240}, margin={"r":0,"t":50,"l":0,"b":0}, showlegend=False, updatemenus=[dict(active=0, buttons=buttons, direction="down", pad={"r": 10, "t": 10}, showactive=True, x=0.01, xanchor="left", y=0.99, yanchor="top")])
    return fig

# --- 3. Streamlit App Layout ---
def main():
    st.set_page_config(page_title="EduReform Nepal Dashboard", layout="wide")
    st.title("🇳🇵 Nepal Education Dashboard")
    st.markdown("An interactive overview of Nepal's educational landscape, based on 2074 B.S. (2017/18 A.D.) data.")

    master_df, faculty_enroll_long, province_dropout, teacher_posts_long, map_data_df, geojson_filename, grade_cols = load_all_data()

    tab1, tab2 = st.tabs(["📊 Provincial Overview", "🗺️ Enrollment Map"])

    with tab1:
        st.header("National Summary")
        national_data = master_df.loc['Nepal (Overall)']
        
        # --- MODIFICATION START: Removed KPIs that had 'nan' values ---
        # Only display the reliable national metric
        col1, col2, col3 = st.columns(3) # Use columns for layout spacing
        with col2: # Center the metric
            st.metric("Total National Higher Ed Students", f"{national_data['total_higher_ed_students']:,.0f}")
        # --- MODIFICATION END ---
        
        st.divider()
        st.header("Comparative Provincial Analysis")
        
        # --- Faculty Enrollment Chart ---
        st.subheader("Higher Education Enrollment by Faculty")
        faculty_data = faculty_enroll_long[faculty_enroll_long['province'] != 'Nepal (Overall)']
        fig_faculty = px.bar(
            faculty_data, 
            x='province', 
            y='students', 
            color='faculty',
            barmode='group',
            title='<b>Higher Ed Enrollment by Faculty and Province</b>',
            labels={'students': 'Number of Students', 'province': 'Province', 'faculty': 'Faculty'},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        st.plotly_chart(fig_faculty, use_container_width=True)

        # --- Dropout Rate Heatmap ---
        st.subheader("Dropout Rates by Grade Level (%)")
        dropout_heatmap_data = province_dropout.set_index('province').drop('Nepal (Overall)', errors='ignore')[grade_cols].T
        fig_dropout = px.imshow(
            dropout_heatmap_data, 
            text_auto='.1f', 
            aspect="auto", 
            color_continuous_scale='Reds',
            title='<b>Provincial Dropout Rate Heatmap</b><br><sup>Darker red indicates a higher dropout rate</sup>',
            labels=dict(x="Province", y="Grade Level", color="Dropout Rate (%)")
        )
        st.plotly_chart(fig_dropout, use_container_width=True)

        # --- Teacher Posts & Training ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Community School Teacher Posts")
            posts_data = teacher_posts_long[teacher_posts_long['province'] != 'Nepal (Overall)']
            fig_posts = px.bar(
                posts_data,
                x='province',
                y='count',
                color='post_type',
                barmode='group',
                title='<b>Approved vs. Rahat Teacher Posts</b>',
                labels={'count': 'Number of Posts', 'province': 'Province', 'post_type': 'Post Type'},
                color_discrete_map={'Approved Posts': COLOR_PALETTE['success'], 'Rahat Posts': COLOR_PALETTE['warning']}
            )
            st.plotly_chart(fig_posts, use_container_width=True)

        with col2:
            st.subheader("Teacher Training Status")
            tt_data = master_df[['full', 'partial', 'untrained_teacher_percent']].drop('Nepal (Overall)')
            fig_teacher_training = px.bar(
                tt_data.reset_index(), 
                x='province', 
                y=['full', 'partial', 'untrained_teacher_percent'], 
                title='<b>Teacher Training Status by Province</b>', 
                labels={'value': 'Percentage of Teachers (%)', 'province': 'Province'}, 
                color_discrete_map={'full': COLOR_PALETTE['success'], 'partial': COLOR_PALETTE['warning'], 'untrained_teacher_percent': COLOR_PALETTE['danger']}
            )
            fig_teacher_training.update_layout(legend_title_text='Training Status')
            st.plotly_chart(fig_teacher_training, use_container_width=True)

        # --- STR and Gender Parity ---
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Student-Teacher Ratio (STR)")
            str_data = master_df['avg_str_reported'].drop('Nepal (Overall)').sort_values()
            
            # --- MODIFICATION START: Added color mapping for STR chart ---
            fig_str = px.bar(
                str_data, 
                x=str_data.index, 
                y=str_data.values, 
                title='<b>Average Student-Teacher Ratio</b>', 
                labels={'y': 'Students per Teacher', 'x': 'Province'}, 
                text_auto='.1f',
                color=str_data.index, # Use province names for color
                color_discrete_sequence=px.colors.qualitative.Vivid # Use a distinct color palette
            )
            fig_str.update_layout(showlegend=False) # Hide the legend as colors are self-explanatory by axis labels
            # --- MODIFICATION END ---
            st.plotly_chart(fig_str, use_container_width=True)

        with col4:
            st.subheader("Gender Parity in Secondary GER")
            ger_data = master_df.drop('Nepal (Overall)')
            fig_gender = go.Figure()
            fig_gender.add_trace(go.Scatter(x=ger_data['ger_sec_boys'], y=ger_data['ger_sec_girls'], mode='markers+text', text=ger_data.index, textposition='top right', marker=dict(size=12, color=px.colors.qualitative.Vivid), name='Provinces'))
            max_val = ger_data[['ger_sec_boys', 'ger_sec_girls']].max().max() + 5
            fig_gender.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', line=dict(dash='dash', color=COLOR_PALETTE['danger']), name='Perfect Parity'))
            fig_gender.update_layout(title='<b>Gender Parity in Secondary GER (9-10)</b>', xaxis_title="Boys' GER (%)", yaxis_title="Girls' GER (%)", showlegend=False)
            st.plotly_chart(fig_gender, use_container_width=True)


    with tab2:
        st.header("National Student Enrollment Map")
        st.markdown("Visualize student enrollment figures across all provinces. Use the dropdown on the map to switch between educational levels.")
        fig_map_enrollment = generate_enrollment_map(map_data_df, geojson_filename)
        # Increased height for better visibility on the map tab
        st.plotly_chart(fig_map_enrollment, use_container_width=True)


if __name__ == '__main__':
    main()