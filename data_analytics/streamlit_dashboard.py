"""
GrainWise Analytics Dashboard
A Streamlit dashboard for analyzing rice grain morphological features and defects.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Page Configuration
st.set_page_config(
    page_title='GrainWise Analytics Dashboard',
    layout='wide',
    page_icon='🌾'
)

# Constants
VARIETIES = ['swarna', '1010', '1001', 'ganga_kaveri', 'mansuri', 'golden_mansuri', 'nati_mansuri', 'sonam', 'ir64']
DEFECTS = ['whole', 'broken', 'chalky', 'damaged', 'discolored', 'foreign']
GRADES = ['Grade A', 'Grade B', 'Common', 'Rejected']
FEATURES = [
    'length_mm', 'width_mm', 'aspect_ratio', 'area_mm2',
    'circularity', 'solidity', 'eccentricity',
    'rg_ratio', 'rb_ratio', 'gb_ratio', 'norm_s'
]

@st.cache_data
def generate_synthetic_data(n=1000):
    """Generates synthetic rice grain data for the dashboard."""
    np.random.seed(42)
    data = []
    
    for _ in range(n):
        variety = np.random.choice(VARIETIES)
        defect_type = np.random.choice(DEFECTS, p=[0.6, 0.1, 0.1, 0.05, 0.1, 0.05])
        
        # Base features depending on variety
        if variety == 'sonam':
            length_mm = np.random.normal(6.5, 0.5)
            width_mm = np.random.normal(1.8, 0.2)
        elif variety == 'swarna':
            length_mm = np.random.normal(4.5, 0.4)
            width_mm = np.random.normal(2.2, 0.2)
        elif variety == 'ir64':
            length_mm = np.random.normal(6.0, 0.4)
            width_mm = np.random.normal(2.0, 0.2)
        else:
            length_mm = np.random.normal(5.5, 0.6)
            width_mm = np.random.normal(2.1, 0.3)
            
        aspect_ratio = length_mm / width_mm
        area_mm2 = length_mm * width_mm * 0.8  # Approximation
        circularity = np.random.normal(0.6, 0.1)
        solidity = np.random.normal(0.95, 0.02)
        eccentricity = np.random.normal(0.8, 0.05)
        
        rg_ratio = np.random.normal(1.1, 0.05)
        rb_ratio = np.random.normal(1.2, 0.06)
        gb_ratio = np.random.normal(1.05, 0.04)
        norm_s = np.random.normal(0.5, 0.1)
        
        # Assign Grade based on defect type
        if defect_type == 'whole':
            grade = 'Grade A'
        elif defect_type in ['chalky', 'discolored']:
            grade = 'Grade B'
        elif defect_type == 'broken':
            grade = 'Common'
        else:
            grade = 'Rejected'
            
        data.append([
            variety, defect_type, length_mm, width_mm, aspect_ratio, area_mm2,
            circularity, solidity, eccentricity, rg_ratio, rb_ratio, gb_ratio, norm_s, grade
        ])
        
    cols = ['variety', 'defect_type'] + FEATURES + ['quality_grade']
    return pd.DataFrame(data, columns=cols)

@st.cache_data
def load_data():
    """Loads grain feature data from CSV or generates synthetic data if not found."""
    file_path = 'data_analytics/data/grain_features.csv'
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.warning(f"Error loading CSV: {e}. Falling back to synthetic data.")
            return generate_synthetic_data()
    else:
        return generate_synthetic_data()

def main():
    st.title("🌾 GrainWise Analytics Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Filters")
    
    # Reset filters button
    if st.sidebar.button("Reset Filters"):
        st.session_state.variety_filter = VARIETIES
        st.session_state.defect_filter = DEFECTS
        st.session_state.grade_filter = 'All'
        st.rerun()
    
    if 'variety_filter' not in st.session_state:
        st.session_state.variety_filter = VARIETIES
    if 'defect_filter' not in st.session_state:
        st.session_state.defect_filter = DEFECTS
    if 'grade_filter' not in st.session_state:
        st.session_state.grade_filter = 'All'
    
    selected_varieties = st.sidebar.multiselect(
        "Select Variety",
        options=VARIETIES,
        default=st.session_state.variety_filter,
        key='variety_filter'
    )
    
    selected_defects = st.sidebar.multiselect(
        "Select Defect Type",
        options=DEFECTS,
        default=st.session_state.defect_filter,
        key='defect_filter'
    )
    
    selected_grade = st.sidebar.radio(
        "Quality Grade",
        options=['All'] + GRADES,
        index=(['All'] + GRADES).index(st.session_state.grade_filter) if st.session_state.grade_filter in ['All'] + GRADES else 0,
        key='grade_filter'
    )
    
    # Apply filters
    filtered_df = df[
        (df['variety'].isin(selected_varieties)) &
        (df['defect_type'].isin(selected_defects))
    ]
    if selected_grade != 'All':
        filtered_df = filtered_df[filtered_df['quality_grade'] == selected_grade]
        
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # KPIs
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    total_grains = len(filtered_df)
    col1.metric("Total Grains", f"{total_grains:,}")
    
    unique_varieties = filtered_df['variety'].nunique()
    col2.metric("Varieties Analysed", unique_varieties)
    
    # Calculate Mean Quality Score
    # 100 - broken% - damaged%*2 - foreign%*3
    if total_grains > 0:
        broken_pct = (filtered_df['defect_type'] == 'broken').mean() * 100
        damaged_pct = (filtered_df['defect_type'] == 'damaged').mean() * 100
        foreign_pct = (filtered_df['defect_type'] == 'foreign').mean() * 100
        quality_score = max(0, 100 - broken_pct - (damaged_pct * 2) - (foreign_pct * 3))
    else:
        quality_score = 0
    col3.metric("Mean Quality Score", f"{quality_score:.1f}")
    
    whole_grain_pct = (filtered_df['defect_type'] == 'whole').mean() * 100 if total_grains > 0 else 0
    col4.metric("Whole Grain %", f"{whole_grain_pct:.1f}%")
    
    st.divider()
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Defect Distribution")
        defect_counts = filtered_df['defect_type'].value_counts().reset_index()
        defect_counts.columns = ['Defect Type', 'Count']
        fig_defect = px.pie(
            defect_counts, 
            names='Defect Type', 
            values='Count',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_defect.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_defect, use_container_width=True)
        
    with col2:
        st.subheader("Grain Count by Variety")
        variety_counts = filtered_df['variety'].value_counts().reset_index()
        variety_counts.columns = ['Variety', 'Count']
        fig_variety = px.bar(
            variety_counts,
            x='Variety',
            y='Count',
            color='Variety',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_variety.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_variety, use_container_width=True)
        
    st.divider()
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Morphological Features")
        selected_feature = st.selectbox("Select Feature for Box Plot", options=FEATURES)
        fig_box = px.box(
            filtered_df,
            x='variety',
            y=selected_feature,
            color='variety',
            points="all",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_box.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col2:
        st.subheader("Length vs Width")
        fig_scatter = px.scatter(
            filtered_df,
            x='length_mm',
            y='width_mm',
            color='variety',
            hover_data=['defect_type', 'quality_grade'],
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_scatter.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.divider()
    
    # Data Table
    st.subheader("Data Explorer")
    st.write(f"Showing {filtered_df.shape[0]} rows and {filtered_df.shape[1]} columns.")
    st.dataframe(filtered_df, use_container_width=True)
    
    # CSV Export
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='grain_filtered_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()
