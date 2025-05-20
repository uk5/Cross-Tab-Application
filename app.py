import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
import io
import json
import os
import base64
from datetime import datetime

# ====================== APP CONFIGURATION ======================
st.set_page_config(
    page_title="Cross-Tabulation Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== STYLING ======================
st.markdown("""
    <style>
    .main {padding: 1rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e89ae;
        color: white;
    }
    .stButton>button {
        background-color: #4e89ae;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    div.block-container {padding-top: 2rem;}
    .dataframe-container {
        height: 400px;
        overflow-y: auto;
    }
    .download-buttons {
        display: flex;
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== SESSION STATE INITIALIZATION ======================
def initialize_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'filtered_data' not in st.session_state:
        st.session_state.filtered_data = None
    if 'tabs' not in st.session_state:
        st.session_state.tabs = [
            {
                'id': str(uuid.uuid4()),
                'name': 'Demographic Analysis',
                'crosstabs': [
                    {
                        'id': str(uuid.uuid4()),
                        'x_axis': 'Gender',
                        'y_axis': 'Age',
                        'visualization': 'Heatmap'
                    }
                ]
            }
        ]
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
    if 'filter_columns' not in st.session_state:
        st.session_state.filter_columns = []
    if 'filter_values' not in st.session_state:
        st.session_state.filter_values = {}
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    if 'rerun_trigger' not in st.session_state:
        st.session_state.rerun_trigger = 0
    if 'config_name' not in st.session_state:
        st.session_state.config_name = "New Configuration"
    if 'data_filename' not in st.session_state:
        st.session_state.data_filename = None
    if 'chart_figures' not in st.session_state:
        st.session_state.chart_figures = {}

    # Create a directory for saved configurations if it doesn't exist
    if not os.path.exists('saved_configs'):
        try:
            os.makedirs('saved_configs')
        except:
            pass  # Handle case where directory creation fails

# ====================== UTILITY FUNCTIONS ======================
def trigger_rerun():
    """Force Streamlit to rerun the app by incrementing a counter"""
    st.session_state.rerun_trigger += 1

@st.cache_data
def create_crosstab(df, x_axis, y_axis, agg_value, refresh_counter=0):
    """Create a crosstab with caching for performance"""
    try:
        if len(df) == 0:
            return pd.DataFrame()
            
        if agg_value == "Count":
            # Simple count crosstab
            ct = pd.crosstab(
                index=df[y_axis], 
                columns=df[x_axis], 
                margins=True, 
                margins_name="Total"
            )
        elif agg_value == "Percentage":
            # Percentage crosstab
            ct = pd.crosstab(
                index=df[y_axis], 
                columns=df[x_axis], 
                normalize='all',
                margins=True, 
                margins_name="Total"
            ) * 100
            ct = ct.round(1)
        else:
            # Aggregation based on numeric column
            ct = pd.pivot_table(
                data=df,
                index=y_axis,
                columns=x_axis,
                values=agg_value,
                aggfunc='mean',
                margins=True,
                margins_name="Total"
            )
            ct = ct.round(1)
        return ct
    except Exception as e:
        print(f"Error in create_crosstab: {e}")
        return pd.DataFrame()

def get_plotly_fig_as_svg(fig):
    """Convert a Plotly figure to SVG for download"""
    return fig.to_image(format="svg")

def get_download_link(data, filename, text, mime_type):
    """Create a download link for any data type"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{text}</a>'
    return href

# ====================== CONFIGURATION MANAGEMENT ======================
def save_configuration(config_name, tabs_config, data_filename=None):
    """Save the current tabs and crosstabs configuration to a JSON file"""
    if not config_name:
        config_name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Prepare the configuration data
    config_data = {
        'tabs': tabs_config,
        'data_filename': data_filename,
        'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to a JSON file
    file_path = os.path.join('saved_configs', f"{config_name.replace(' ', '_')}.json")
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    return file_path

def load_configuration(file_path):
    """Load a saved configuration from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        
        return config_data
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

def get_saved_configs():
    """Get a list of all saved configurations"""
    try:
        if not os.path.exists('saved_configs'):
            return []
        
        config_files = [f for f in os.listdir('saved_configs') if f.endswith('.json')]
        configs = []
        
        for config_file in config_files:
            try:
                file_path = os.path.join('saved_configs', config_file)
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Extract config name from filename
                config_name = config_file.replace('.json', '').replace('_', ' ')
                
                configs.append({
                    'name': config_name,
                    'path': file_path,
                    'saved_date': config_data.get('saved_date', 'Unknown'),
                    'data_filename': config_data.get('data_filename', 'Unknown')
                })
            except:
                # Skip files that can't be loaded
                pass
        
        return configs
    except:
        return []

# ====================== DATA LOADING ======================
def load_data_file(uploaded_file):
    """Load data from uploaded file and handle data type conversions"""
    try:
        # Store the filename in session state
        st.session_state.data_filename = uploaded_file.name
        
        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            # Read CSV with all columns as strings to avoid mixed type issues
            data = pd.read_csv(uploaded_file, dtype=str)
        else:  # .xls or .xlsx
            # Read Excel with all columns as strings to avoid mixed type issues
            data = pd.read_excel(uploaded_file, dtype=str)
        
        # Convert numeric columns back to numeric where appropriate
        for col in data.columns:
            # Try to convert to numeric, but keep as string if it fails
            try:
                # If column can be converted cleanly to numeric, do so
                if pd.to_numeric(data[col], errors='coerce').notna().all():
                    data[col] = pd.to_numeric(data[col])
            except:
                pass  # Keep as string if conversion fails
        
        st.session_state.data = data
        st.session_state.filtered_data = data.copy()
        st.success(f"Successfully loaded {uploaded_file.name} with {len(data)} records and {len(data.columns)} columns")
        return True
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return False

# ====================== VISUALIZATION FUNCTIONS ======================
def render_visualization(visualization, ct, x_axis, y_axis, agg_value, crosstab_id):
    """Render the appropriate visualization based on the selection"""
    if len(ct) <= 0:
        st.warning("No data available for this cross-tabulation with current filters.")
        return

    # Create a container to hold the visualization
    viz_container = st.container()
    
    with viz_container:
        chart_fig = None  # Initialize figure variable for SVG export
        
        if visualization == "Table":
            st.dataframe(ct, use_container_width=True)
            # No chart to save as SVG for tables
        
        elif visualization == "Heatmap":
            try:
                if len(ct) > 1 and len(ct.columns) > 1:  # Make sure we have data
                    ct_heatmap = ct.iloc[:-1, :-1]
                    
                    fig = px.imshow(
                        ct_heatmap,
                        labels=dict(x=x_axis, y=y_axis, color="Value"),
                        x=ct_heatmap.columns,
                        y=ct_heatmap.index,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Blues"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True, key=f"heatmap_{crosstab_id}")
                    chart_fig = fig  # Save figure for SVG export
                else:
                    st.warning("Not enough data for heatmap visualization.")
                    st.dataframe(ct, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating heatmap: {e}")
                st.dataframe(ct, use_container_width=True)
        
        elif visualization == "Bar Chart":
            try:
                if len(ct) > 1 and len(ct.columns) > 1:  # Make sure we have data
                    ct_bar = ct.iloc[:-1, :-1]  # Remove totals
                    fig = px.bar(
                        ct_bar,
                        barmode='group',
                        labels={'value': agg_value, 'index': y_axis},
                        title=f"{y_axis} by {x_axis}"
                    )
                    fig.update_layout(height=500, xaxis_title=y_axis, yaxis_title=agg_value)
                    st.plotly_chart(fig, use_container_width=True, key=f"barchart_{crosstab_id}")
                    chart_fig = fig  # Save figure for SVG export
                else:
                    st.warning("Not enough data for bar chart visualization.")
                    st.dataframe(ct, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating bar chart: {e}")
                st.dataframe(ct, use_container_width=True)
        
        elif visualization == "Stacked Bar":
            try:
                if len(ct) > 1 and len(ct.columns) > 1:  # Make sure we have data
                    ct_stacked = ct.iloc[:-1, :-1]  # Remove totals
                    fig = px.bar(
                        ct_stacked,
                        barmode='stack',
                        labels={'value': agg_value, 'index': y_axis},
                        title=f"{y_axis} by {x_axis}"
                    )
                    fig.update_layout(height=500, xaxis_title=y_axis, yaxis_title=agg_value)
                    st.plotly_chart(fig, use_container_width=True, key=f"stackedbar_{crosstab_id}")
                    chart_fig = fig  # Save figure for SVG export
                else:
                    st.warning("Not enough data for stacked bar visualization.")
                    st.dataframe(ct, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating stacked bar chart: {e}")
                st.dataframe(ct, use_container_width=True)
        
        # Store the figure in session state for download
        if chart_fig is not None:
            st.session_state.chart_figures[crosstab_id] = chart_fig
        
        # Create download buttons container
        download_col1, download_col2, download_col3 = st.columns(3)
        
        with download_col1:
            # CSV download button
            download_ct_csv = ct.to_csv().encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=download_ct_csv,
                file_name=f"crosstab_{x_axis}_{y_axis}.csv",
                mime="text/csv",
                key=f"download_csv_{crosstab_id}"
            )
        
        with download_col2:
            # Excel download button
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                ct.to_excel(writer, sheet_name='Data')
                # Removed writer.close() - it's automatically closed by the context manager
            buffer.seek(0)
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name=f"crosstab_{x_axis}_{y_axis}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_{crosstab_id}"
            )
        
        with download_col3:
            # SVG download button (only for charts, not tables)
            if chart_fig is not None and visualization != "Table":
                try:
                    svg_data = chart_fig.to_image(format="svg")
                    st.download_button(
                        label="Download as SVG",
                        data=svg_data,
                        file_name=f"chart_{x_axis}_{y_axis}.svg",
                        mime="image/svg+xml",
                        key=f"download_svg_{crosstab_id}"
                    )
                except Exception as e:
                    st.error(f"Error creating SVG: {e}")

# ====================== UI COMPONENTS ======================
def render_configuration_management():
    """Render the configuration management UI"""
    st.subheader("Configuration")
    config_tab1, config_tab2 = st.tabs(["Save", "Load"])
    
    with config_tab1:
        st.session_state.config_name = st.text_input("Configuration Name", value=st.session_state.config_name)
        if st.button("Save Current Configuration"):
            if st.session_state.tabs:
                file_path = save_configuration(
                    st.session_state.config_name, 
                    st.session_state.tabs,
                    st.session_state.data_filename
                )
                st.success(f"Configuration saved to: {file_path}")
            else:
                st.warning("No tabs to save. Please create at least one tab first.")
    
    with config_tab2:
        saved_configs = get_saved_configs()
        if saved_configs:
            selected_config = st.selectbox(
                "Select a saved configuration",
                options=[config['name'] for config in saved_configs],
                format_func=lambda x: f"{x} (Saved: {next((c['saved_date'] for c in saved_configs if c['name'] == x), 'Unknown')})",
                key="config_select_box"
            )
            
            selected_config_data = next((c for c in saved_configs if c['name'] == selected_config), None)
            
            if selected_config_data and st.button("Load Selected Configuration"):
                config_data = load_configuration(selected_config_data['path'])
                if config_data:
                    st.session_state.tabs = config_data['tabs']
                    st.session_state.config_name = selected_config
                    st.success(f"Configuration '{selected_config}' loaded successfully!")
                    
                    # Check if data file matches
                    if config_data.get('data_filename') and config_data['data_filename'] != st.session_state.data_filename:
                        st.info(f"Note: This configuration was created with a different data file: {config_data['data_filename']}")
                    
                    trigger_rerun()
        else:
            st.info("No saved configurations found. Create and save a configuration first.")

def render_data_filters():
    """Render the data filtering UI"""
    st.subheader("Filter Data")
    filter_cols = st.multiselect("Select columns to filter by", options=st.session_state.data.columns.tolist(), key="filter_col_select")
    
    # Clear previous filter values if columns changed
    if filter_cols != st.session_state.filter_columns:
        st.session_state.filter_values = {}
        st.session_state.filter_columns = filter_cols
    
    # Create filter inputs for selected columns
    filter_col1, filter_col2 = st.columns(2)
    
    filter_applied = False
    for i, col in enumerate(filter_cols):
        with filter_col1 if i % 2 == 0 else filter_col2:
            if st.session_state.data[col].dtype == 'object' or st.session_state.data[col].nunique() < 15:
                # Categorical column - use multiselect
                options = ["All"] + sorted(st.session_state.data[col].dropna().unique().tolist())
                selected = st.multiselect(
                    f"Filter by {col}",
                    options=options,
                    default=["All"] if col not in st.session_state.filter_values else st.session_state.filter_values[col],
                    key=f"filter_{col}"
                )
                if selected and "All" not in selected:
                    st.session_state.filter_values[col] = selected
                    filter_applied = True
                else:
                    # "All" was selected or nothing was selected
                    if col in st.session_state.filter_values:
                        del st.session_state.filter_values[col]
            else:
                # Numeric column - use slider
                min_val = float(st.session_state.data[col].min())
                max_val = float(st.session_state.data[col].max())
                
                if col not in st.session_state.filter_values:
                    default_values = (min_val, max_val)
                else:
                    default_values = st.session_state.filter_values[col]
                
                values = st.slider(
                    f"Filter by {col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_values,
                    key=f"filter_{col}"
                )
                
                if values != (min_val, max_val):
                    st.session_state.filter_values[col] = values
                    filter_applied = True
                else:
                    if col in st.session_state.filter_values:
                        del st.session_state.filter_values[col]
    
    # Apply filters
    filtered_df = st.session_state.data.copy()
    for col, values in st.session_state.filter_values.items():
        if isinstance(values, list):  # Categorical filter
            filtered_df = filtered_df[filtered_df[col].isin(values)]
        else:  # Numeric range filter
            filtered_df = filtered_df[(filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])]
    
    st.session_state.filtered_data = filtered_df
    
    # Reset filters button
    if filter_applied:
        if st.button("Reset All Filters"):
            st.session_state.filter_values = {}
            st.session_state.filtered_data = st.session_state.data.copy()
            trigger_rerun()
    
    # Display filtered data info
    st.info(f"Showing {len(filtered_df)} of {len(st.session_state.data)} records")
    
    return filter_applied

def render_export_options():
    """Render the export options UI"""
    # Choose what to export
    export_option = st.radio(
        "What data would you like to export?",
        options=["All Data", "Filtered Data"],
        horizontal=True,
        key="export_option_radio"
    )
    
    export_data = st.session_state.data if export_option == "All Data" else st.session_state.filtered_data
    
    # Choose export format
    export_format = st.selectbox(
        "Select export format",
        options=["CSV", "Excel (.xlsx)"],
        key="export_format_select"
    )
    
    # Export button
    col1, col2 = st.columns([1, 3])
    with col1:
        if export_format == "CSV":
            csv_data = export_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Data",
                data=csv_data,
                file_name="data_export.csv",
                mime="text/csv",
                key="export_data_csv"
            )
        else:  # Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_data.to_excel(writer, sheet_name='Data', index=False)
                # Removed writer.close() - it's automatically closed by the context manager
            buffer.seek(0)
            excel_data = buffer.getvalue()
            st.download_button(
                label="Download Data",
                data=excel_data,
                file_name="data_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_data_excel"
            )
    
    # Preview of data being exported
    st.subheader("Export Preview")
    st.dataframe(export_data.head(10), use_container_width=True)
    st.info(f"Exporting {len(export_data)} rows and {len(export_data.columns)} columns")

def render_config_export():
    """Render the configuration export UI"""
    st.subheader("Export Configuration")
    st.info("You can export your current tab and crosstab configuration to share with others.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Export Configuration", key="export_config_button"):
            # Create a JSON representation of the configuration
            config_data = {
                'tabs': st.session_state.tabs,
                'data_filename': st.session_state.data_filename,
                'saved_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Convert to JSON and prepare for download
            config_json = json.dumps(config_data, indent=4)
            config_bytes = config_json.encode('utf-8')
            
            # Create a download button
            st.download_button(
                label="Download Configuration File",
                data=config_bytes,
                file_name=f"{st.session_state.config_name.replace(' ', '_')}.json",
                mime="application/json",
                key="download_config_button"
            )

def render_crosstab_ui(tab_index, tab):
    """Render the UI for a single tab of crosstabs"""
    st.subheader(f"{tab['name']}")
    
    # Tab control buttons
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Add New Crosstab", key=f"add_crosstab_{tab['id']}"):
            st.session_state.tabs[tab_index]['crosstabs'].append({
                'id': str(uuid.uuid4()),
                'x_axis': st.session_state.data.columns[0] if len(st.session_state.data.columns) > 0 else 'Column',
                'y_axis': st.session_state.data.columns[1] if len(st.session_state.data.columns) > 1 else 'Column',
                'visualization': 'Heatmap'
            })
            trigger_rerun()
    
    # Process each crosstab
    for j, crosstab in enumerate(tab['crosstabs']):
        st.markdown(f"#### Crosstab {j+1}")
        
        # Crosstab controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        # Get categorical columns
        categorical_cols = [col for col in st.session_state.data.columns if 
                           st.session_state.data[col].dtype == 'object' or 
                           st.session_state.data[col].nunique() < 15]
        
        if not categorical_cols:
            categorical_cols = list(st.session_state.data.columns)
        
        with col1:
            # Select X-axis
            x_axis_index = 0
            if crosstab['x_axis'] in categorical_cols:
                x_axis_index = categorical_cols.index(crosstab['x_axis'])
            
            x_axis = st.selectbox(
                "Select X-Axis (Columns)",
                options=categorical_cols,
                index=x_axis_index,
                key=f"x_axis_{crosstab['id']}"
            )
            st.session_state.tabs[tab_index]['crosstabs'][j]['x_axis'] = x_axis
        
        with col2:
            # Select Y-axis
            y_axis_index = 0
            if crosstab['y_axis'] in categorical_cols:
                y_axis_index = categorical_cols.index(crosstab['y_axis'])
                if y_axis_index == x_axis_index and len(categorical_cols) > 1:
                    y_axis_index = (y_axis_index + 1) % len(categorical_cols)
            
            y_axis = st.selectbox(
                "Select Y-Axis (Rows)",
                options=categorical_cols,
                index=y_axis_index,
                key=f"y_axis_{crosstab['id']}"
            )
            st.session_state.tabs[tab_index]['crosstabs'][j]['y_axis'] = y_axis
        
        with col3:
            # Select aggregation method
            numeric_cols = [col for col in st.session_state.data.columns if col not in categorical_cols]
            agg_options = ["Count", "Percentage"] + numeric_cols
            
            agg_value = st.selectbox(
                "Aggregation Method",
                options=agg_options,
                key=f"agg_{crosstab['id']}"
            )
        
        with col4:
            # Select visualization type
            viz_options = ["Table", "Heatmap", "Bar Chart", "Stacked Bar"]
            
            visualization = st.selectbox(
                "Visualization Type",
                options=viz_options,
                index=viz_options.index(crosstab['visualization']) if crosstab['visualization'] in viz_options else 0,
                key=f"viz_{crosstab['id']}"
            )
            st.session_state.tabs[tab_index]['crosstabs'][j]['visualization'] = visualization
        
        # Create crosstab using filtered data and the cached function
        refresh_key = st.session_state.refresh_counter  # Use as cache-busting key
        ct = create_crosstab(
            st.session_state.filtered_data, 
            x_axis, 
            y_axis, 
            agg_value,
            refresh_key
        )
        
        # Render the selected visualization
        render_visualization(visualization, ct, x_axis, y_axis, agg_value, crosstab['id'])
        
        # Delete crosstab button (only if more than one exists)
        if len(tab['crosstabs']) > 1:
            if st.button("Delete this Crosstab", key=f"delete_{crosstab['id']}"):
                st.session_state.tabs[tab_index]['crosstabs'] = [ct for ct in tab['crosstabs'] if ct['id'] != crosstab['id']]
                trigger_rerun()
        
        st.markdown("---")
    
    # Delete tab option
    if len(st.session_state.tabs) > 1 and st.button(f"Delete '{tab['name']}' Tab", key=f"delete_tab_{tab['id']}"):
        st.session_state.tabs = [t for t in st.session_state.tabs if t['id'] != tab['id']]
        trigger_rerun()

def render_data_editor():
    """Render the data editor UI"""
    # Toggle edit mode
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.session_state.edit_mode = st.toggle("Enable Edit Mode", st.session_state.edit_mode, key="edit_mode_toggle")
    with col2:
        st.session_state.auto_refresh = st.toggle("Auto-Refresh", st.session_state.auto_refresh, key="auto_refresh_toggle")
    with col3:
        if st.session_state.edit_mode:
            st.info("Edit mode enabled. Click on any cell to modify its value.")
            if not st.session_state.auto_refresh:
                st.warning("Auto-refresh is disabled. Use the 'Update Crosstabs' button after making changes.")
    
    # Manually trigger refresh for crosstabs
    if not st.session_state.auto_refresh:
        if st.button("Update Crosstabs", key="update_crosstabs_button"):
            st.session_state.refresh_counter += 1
            # Force a redraw
            trigger_rerun()
    
    # Render data filters
    filter_applied = render_data_filters()
    
    # Make dataframe editable if edit mode is on
    if st.session_state.edit_mode:
        edited_df = st.data_editor(
            st.session_state.filtered_data,
            use_container_width=True,
            num_rows="dynamic",
            height=400,
            key="data_editor"
        )
        
        # Update the main dataframe with edits
        if edited_df is not None:
            # Only update if there are actual changes
            if not st.session_state.filtered_data.equals(edited_df):
                # Get indices of filtered data
                filtered_indices = st.session_state.filtered_data.index.tolist()
                
                # Update the main dataframe efficiently
                if len(filtered_indices) > 0:
                    # Create a mask for the indices we want to update
                    update_mask = st.session_state.data.index.isin(filtered_indices)
                    
                    # Only update columns that have changed
                    changed_cols = [col for col in st.session_state.data.columns 
                                  if not st.session_state.filtered_data[col].equals(edited_df[col])]
                    
                    if changed_cols:
                        # Update only the necessary columns at the filtered indices
                        for col in changed_cols:
                            st.session_state.data.loc[update_mask, col] = edited_df.loc[filtered_indices, col].values
                
                # Update the filtered data
                st.session_state.filtered_data = edited_df
                
                # Only show success message and trigger refresh if auto-refresh is on
                if st.session_state.auto_refresh:
                    st.success("Data updated successfully!")
                    st.session_state.refresh_counter += 1  # Trigger refresh of crosstabs
    else:
        # Just display the dataframe without editing capabilities
        st.dataframe(st.session_state.filtered_data, use_container_width=True, height=400)

def render_cross_tabulation():
    """Render the cross tabulation UI"""
    # Tab management
    col1, col2 = st.columns([3, 1])
    with col1:
        new_tab_name = st.text_input("New Tab Name", key="new_tab_name_input")
    with col2:
        if st.button("Add New Tab", key="add_new_tab_button") and new_tab_name:
            st.session_state.tabs.append({
                'id': str(uuid.uuid4()),
                'name': new_tab_name,
                'crosstabs': [
                    {
                        'id': str(uuid.uuid4()),
                        'x_axis': st.session_state.data.columns[0] if len(st.session_state.data.columns) > 0 else 'Column',
                        'y_axis': st.session_state.data.columns[1] if len(st.session_state.data.columns) > 1 else 'Column',
                        'visualization': 'Heatmap'
                    }
                ]
            })
            trigger_rerun()
    
    # Create tabs for cross-tabulation
    tab_names = [tab['name'] for tab in st.session_state.tabs]
    tabs = st.tabs(tab_names)
    
    # Process each tab
    for i, tab in enumerate(st.session_state.tabs):
        with tabs[i]:
            render_crosstab_ui(i, tab)

def render_export_tab():
    """Render the export tab UI"""
    # Data export
    render_export_options()
    
    # Configuration export
    render_config_export()

def render_no_data_ui():
    """Render the UI when no data is loaded"""
    st.info("Please upload a data file (CSV, XLS, or XLSX) to get started.")
    
    # Show saved configurations even if no data is loaded
    saved_configs = get_saved_configs()
    if saved_configs:
        st.subheader("Load a Saved Configuration")
        selected_config = st.selectbox(
            "Select a saved configuration",
            options=[config['name'] for config in saved_configs],
            format_func=lambda x: f"{x} (Saved: {next((c['saved_date'] for c in saved_configs if c['name'] == x), 'Unknown')})",
            key="no_data_config_select"  # Added unique key here
        )
        
        selected_config_data = next((c for c in saved_configs if c['name'] == selected_config), None)
        
        if selected_config_data and st.button("Load Selected Configuration", key="load_config_no_data"):
            config_data = load_configuration(selected_config_data['path'])
            if config_data:
                st.session_state.tabs = config_data['tabs']
                st.session_state.config_name = selected_config
                st.success(f"Configuration '{selected_config}' loaded successfully!")
                st.info(f"Please upload the data file: {config_data.get('data_filename', 'Unknown')}")
                trigger_rerun()
    
    st.markdown("""
    ### How to use this app:
    
    1. Upload your data file using the file uploader above
    2. Use the Data Editor tab to view and edit your data
    3. Create cross-tabulations in the Cross-Tabulation tab
    4. Export your modified data in the Export Data tab
    5. Save your configuration to continue working on it later
    
    This app allows you to:
    - Edit your data directly and see changes reflected in cross-tabulations
    - Create multiple tabs with different cross-tabulations
    - Visualize your data in various formats (tables, heatmaps, bar charts)
    - Filter your data dynamically
    - Export your modified data in CSV or Excel format
    - Export charts as SVG files
    - Save and load tab configurations to continue your analysis later
    """)

# ====================== MAIN APP ======================
def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App title
    st.title("Cross-Tabulation Analysis Tool")
    st.markdown("### Interactive Data Analysis & Visualization")
    
    # File upload and configuration management
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader("Upload CSV, XLS, or XLSX file", type=['csv', 'xls', 'xlsx'], key="file_uploader")
        if uploaded_file is not None:
            load_data_file(uploaded_file)
    
    with col2:
        # Configuration management
        render_configuration_management()
    
    # Render main content based on whether data is loaded
    if st.session_state.data is not None:
        # Tab selection for main interface
        main_tabs = st.tabs(["Data Editor", "Cross-Tabulation", "Export Data"])
        
        # Tab 1: Data Editor
        with main_tabs[0]:
            st.header("Data Editor")
            render_data_editor()
        
        # Tab 2: Cross-Tabulation
        with main_tabs[1]:
            st.header("Cross-Tabulation")
            render_cross_tabulation()
        
        # Tab 3: Export Data
        with main_tabs[2]:
            st.header("Export Data")
            render_export_tab()
    else:
        # Show UI for when no data is loaded
        render_no_data_ui()

# Run the app
if __name__ == "__main__":
    main()
