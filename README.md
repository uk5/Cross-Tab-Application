# ﻿Cross-Tabulation Analysis Tool
An interactive data analysis and visualization tool built with Streamlit that allows users to create, customize, and export cross-tabulations from tabular data._



Features
- Dynamic Data Loading: Upload CSV, XLS, or XLSX files for analysis
- Interactive Data Editing: Edit your data directly in the application
- Powerful Cross-Tabulation: Create multiple cross-tabs with different axes and aggregations
- Visualization Options: View your data as tables, heatmaps, bar charts, or stacked bars
- Export Capabilities: Export tables as CSV/Excel and charts as SVG
- Configuration Management: Save and load your analysis configurations
- Data Filtering: Apply filters to focus on specific subsets of your data
- Multi-tab Interface: Organize your analyses in separate tabs

# Installation

#Local Installation

#Clone the repository:
 - bash
 -git clone https://github.com/uk5/cross-tabulation-tool.git
 - cd cross-tabulation-tool

# Install dependencies:
 - bash
 - pip install -r requirements.txt
- Run the app:
- bash
- streamlit run app.py
- Requirements
- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Plotly
- XlsxWriter
- Usage Guide
- Loading Data

#Start the application and use the file uploader to select your data file (CSV, XLS, or XLSX).
The app will automatically detect and convert appropriate columns to numeric types.
Editing Data
Navigate to the "Data Editor" tab.
Toggle "Enable Edit Mode" to make changes to your data.
Use the "Auto-Refresh" option to see changes reflected in visualizations immediately.
Filter data using column selectors and value filters.
Creating Cross-Tabulations
Navigate to the "Cross-Tabulation" tab.
Create new tabs to organize different analyses.
For each tab:
Add cross-tabulations by clicking "Add New Crosstab"
Select X-axis and Y-axis variables
Choose aggregation method (Count, Percentage, or numeric variable)
Select visualization type (Table, Heatmap, Bar Chart, Stacked Bar)
Exporting Data
Navigate to the "Export Data" tab.
Choose to export all data or filtered data.
Select format (CSV or Excel).
Click "Download Data" to save to your local machine.
For charts, use the "Download as SVG" option directly below each visualization.
Saving & Loading Configurations
Use the "Save" tab in the Configuration section to save your current setup.
Use the "Load" tab to restore previously saved configurations.
Export configuration files to share with team members.
Deployment
This application can be deployed to Streamlit Cloud for team access:

Push the code to GitHub:
bash
git add .
git commit -m "Your commit message"
git push
Visit Streamlit Cloud and sign in with GitHub.
Click "New app" and select your repository.
Set the main file path to app.py.
Click "Deploy!"
Data Handling
The app creates a saved_configs directory to store configuration JSON files
Data is processed in-memory and not stored on the server
Any edited data can be exported to your local machine
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Built with Streamlit
Visualizations powered by Plotly
Data processing with Pandas
