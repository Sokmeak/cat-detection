"""
CSS Styles Configuration
========================
Consistent color scheme and styling for the application
"""

# Color scheme
COLORS = {
    'primary_dark': '#0F2854',      # Dark blue for headers and text
    'primary_medium': '#1C4D8D',    # Medium blue for secondary elements
    'primary_light': '#4988C4',     # Light blue for buttons and accents
    'accent_light': '#BDE8F5',      # Very light blue for backgrounds
    'success_bg': '#E8F5F1',
    'neutral_bg': '#F5F5F5',
    'sidebar_bg': '#F8FBFD',
}

# CSS styling
CSS_STYLES = """
<style>
    /* Color Variables */
    :root {
        --primary-dark: #0F2854;
        --primary-medium: #1C4D8D;
        --primary-light: #4988C4;
        --accent-light: #BDE8F5;
    }
    
    /* Material Icons */
    .material-icons {
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F2854;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        border-bottom: 3px solid #4988C4;
    }
    
    .main-header .material-icons {
        color: #4988C4;
        margin-right: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: #1C4D8D;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #BDE8F5;
    }
    
    /* Detection Result Cards */
    .detection-result {
        font-size: 1.3rem;
        font-weight: 600;
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1.5rem 0;
        border-left: 5px solid;
    }
    
    .detection-result .material-icons {
        vertical-align: middle;
        margin-right: 0.5rem;
        font-size: 1.5em;
    }
    
    .cat-detected {
        background-color: #E8F5F1;
        color: #0F2854;
        border-left-color: #4988C4;
    }
    
    .cat-detected .material-icons {
        color: #4988C4;
    }
    
    .no-cat {
        background-color: #F5F5F5;
        color: #1C4D8D;
        border-left-color: #BDE8F5;
    }
    
    .no-cat .material-icons {
        color: #1C4D8D;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #BDE8F5;
        color: #0F2854;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #4988C4;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1C4D8D;
        font-weight: 600;
    }
    
    [data-testid="stMetricLabel"] {
        color: #0F2854;
        font-size: 0.9rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FBFD;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #0F2854;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #0F2854;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4988C4;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1C4D8D;
        box-shadow: 0 2px 8px rgba(73, 136, 196, 0.3);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4988C4;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F8FBFD;
        color: #1C4D8D;
        border-radius: 6px 6px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    /* Caption styling */
    .stCaption {
        color: #1C4D8D !important;
        font-size: 0.85rem;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background-color: rgba(189, 232, 245, 0.2);
        border: 1px solid #BDE8F5;
        border-radius: 6px;
    }
    
    [data-testid="stExpander"] summary {
        color: #0F2854;
        font-weight: 500;
    }
    
    /* Info boxes */
    [data-testid="stAlert"] {
        color: #0F2854;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4988C4;
        color: white;
    }
    
    /* Image containers */
    .image-container {
        border: 2px solid #BDE8F5;
        border-radius: 8px;
        padding: 0.5rem;
        background-color: white;
    }
</style>
"""
