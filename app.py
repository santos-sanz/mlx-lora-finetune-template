#!/usr/bin/env python3
"""
MLX LoRA Fine-tuning - Streamlit UI

A modern interface for fine-tuning LLM models using LoRA and MLX on Apple Silicon.
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
import json
import time
import threading
import queue

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, LoRAConfig, TrainingConfig, ModelConfig, DataConfig, OutputConfig, HuggingFaceConfig
from src.data_utils import load_dataset, convert_to_mlx_format, generate_with_openrouter, is_openrouter_configured, get_openrouter_config
from src.hf_utils import get_hf_token, upload_model, upload_checkpoint, list_checkpoints, check_repo_exists


# ============================================================================
# File/Folder Input Helpers
# ============================================================================

def file_picker(label: str, default_path: str = "", file_types: list = None, key: str = None, help: str = None) -> str:
    """
    Simple file path input with validation.
    
    Args:
        label: Input label
        default_path: Default value
        file_types: Accepted extensions for help text
        key: Widget key
        help: Help text
    
    Returns:
        File path string
    """
    picker_key = key or f"fp_{label}"
    
    if picker_key not in st.session_state:
        st.session_state[picker_key] = default_path
    
    help_text = help
    if not help_text and file_types:
        help_text = f"Supported: {', '.join(file_types)}"
    
    path = st.text_input(
        label,
        value=st.session_state[picker_key],
        key=f"{picker_key}_input",
        help=help_text
    )
    
    st.session_state[picker_key] = path
    
    # Show validation
    if path:
        p = Path(path)
        if p.exists():
            if p.is_file():
                st.caption(f"✅ File exists ({p.stat().st_size:,} bytes)")
            else:
                st.caption("⚠️ Path is a directory, not a file")
        else:
            st.caption("⚠️ File not found")
    
    return path


def folder_picker(label: str, default_path: str = "", key: str = None, help: str = None) -> str:
    """
    Simple folder path input with validation.
    
    Args:
        label: Input label
        default_path: Default value
        key: Widget key
        help: Help text
    
    Returns:
        Folder path string
    """
    picker_key = key or f"dp_{label}"
    
    if picker_key not in st.session_state:
        st.session_state[picker_key] = default_path
    
    path = st.text_input(
        label,
        value=st.session_state[picker_key],
        key=f"{picker_key}_input",
        help=help or "Enter folder path"
    )
    
    st.session_state[picker_key] = path
    
    # Show validation
    if path:
        p = Path(path)
        if p.exists():
            if p.is_dir():
                try:
                    count = len(list(p.iterdir()))
                    st.caption(f"✅ Folder exists ({count} items)")
                except:
                    st.caption("✅ Folder exists")
            else:
                st.caption("⚠️ Path is a file, not a folder")
        else:
            st.caption("⚠️ Folder not found (will be created)")
    
    return path


# ============================================================================
# Page Configuration & Custom CSS
# ============================================================================

st.set_page_config(
    page_title="MLX LoRA Fine-tuning",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state early
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_theme_css(theme: str = 'dark') -> str:
    """Generate CSS based on theme selection."""
    
    if theme == 'light':
        # Light mode colors
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    code, pre, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

    /* LIGHT MODE */
    .stApp { background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%); }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border-right: 1px solid rgba(255, 107, 107, 0.2);
    }
    
    /* Collapsed sidebar - light mode */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
    }
    
    /* Sidebar collapse button - light mode */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"] {
        color: #FF6B6B !important;
        background: rgba(255, 107, 107, 0.15) !important;
        border-radius: 8px !important;
    }
    
    /* FORCE VISIBILITY - Light mode collapsed sidebar */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 2px solid rgba(255, 107, 107, 0.3) !important;
        min-width: 3rem !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarNavCollapseIcon"],
    button[kind="headerNoPadding"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #FF6B6B !important;
        background: rgba(255, 107, 107, 0.15) !important;
        border-radius: 8px !important;
        min-width: 2rem !important;
        min-height: 2rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #FF6B6B !important;
        stroke: #FF6B6B !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 1.5rem;
    }
    
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: #1e293b !important; font-weight: 500 !important; }
    
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown td, .stMarkdown th { color: #1e293b !important; }
    .stMarkdown strong, .stMarkdown b { color: #0f172a !important; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 { color: #0f172a !important; }
    
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stSlider label, .stCheckbox label, .stRadio label, .stMultiSelect label {
        color: #0f172a !important; font-weight: 600 !important;
    }
    
    .stCheckbox, .stCheckbox span, .stCheckbox label, .stCheckbox p,
    [data-testid="stCheckbox"], [data-testid="stCheckbox"] span,
    .stRadio, .stRadio span, .stRadio label, .stRadio p,
    [data-testid="stRadio"], [data-testid="stRadio"] span {
        color: #1e293b !important; -webkit-text-fill-color: #1e293b !important;
    }
    
    .stCaption, [data-testid="stCaption"] { color: #64748b !important; }
    
    .streamlit-expanderHeader, .streamlit-expanderHeader p,
    .streamlit-expanderContent, .streamlit-expanderContent p,
    [data-testid="stExpander"] p, [data-testid="stExpander"] span { color: #1e293b !important; }

    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.9); border-radius: 16px;
        border: 1px solid rgba(255, 142, 83, 0.2); padding: 24px;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(241, 245, 249, 0.95); border-radius: 12px; padding: 6px;
    }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; font-weight: 600; }
    .stTabs [data-baseweb="tab"]:hover { background: rgba(255, 142, 83, 0.15); color: #0f172a !important; }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: #ffffff !important; box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }

    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white !important; border: none; border-radius: 10px;
        padding: 12px 28px; font-weight: 700; box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }
    .stButton > button:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(255, 107, 107, 0.4); }

    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stTextArea > div > textarea {
        background: #ffffff !important; border: 1.5px solid #cbd5e1 !important;
        border-radius: 10px !important; color: #1e293b !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > textarea:focus {
        border-color: #FF8E53 !important; box-shadow: 0 0 0 3px rgba(255, 142, 83, 0.2) !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: #ffffff !important; border: 1.5px solid #cbd5e1 !important; color: #1e293b !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(255, 142, 83, 0.2);
        border-radius: 14px; padding: 20px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    [data-testid="stMetricValue"] { color: #0f172a !important; }

    .main-title {
        font-size: 2.8rem; font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 40%, #e85d04 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #64748b !important; }
    .section-header { color: #0f172a !important; }
    .feature-card {
        background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(255, 142, 83, 0.15);
        border-radius: 16px; padding: 28px;
    }
    .feature-title { color: #0f172a !important; }
    .feature-desc { color: #64748b !important; }

    .stInfo { background: rgba(99, 179, 237, 0.1) !important; border: 1px solid rgba(99, 179, 237, 0.3) !important; color: #1e40af !important; }
    .stSuccess { background: rgba(34, 197, 94, 0.1) !important; border: 1px solid rgba(34, 197, 94, 0.3) !important; color: #047857 !important; }
    .stWarning { background: rgba(255, 230, 109, 0.2) !important; border: 1px solid rgba(234, 179, 8, 0.4) !important; color: #92400e !important; }
    .stError { background: rgba(255, 107, 107, 0.1) !important; border: 1px solid rgba(255, 107, 107, 0.3) !important; color: #b91c1c !important; }

    .stCodeBlock { background: #1e293b !important; border-radius: 10px !important; }
    .stCodeBlock code { color: #e2e8f0 !important; }
    .streamlit-expanderHeader { background: rgba(241, 245, 249, 0.9) !important; }
    hr { border-color: rgba(255, 142, 83, 0.2) !important; }
    [data-testid="stTooltipIcon"] { color: #FF8E53 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
"""
    else:
        # Dark mode colors (default)
        return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    * { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }
    code, pre, .stCodeBlock { font-family: 'JetBrains Mono', monospace !important; }

    /* ========================================
       COLOR PALETTE - DARK MODE
       Primary: Coral (#FF6B6B) 
       Secondary: Orange (#FF8E53)
       Accent: Gold (#FFE66D)
       Dark: Slate (#1a1d29)
       ======================================== */

    /* Main background - rich dark gradient */
    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1d29 50%, #141821 100%);
    }
    
    /* Animated background orbs */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(255, 142, 83, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 107, 107, 0.08) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(255, 230, 109, 0.04) 0%, transparent 40%);
        animation: float 20s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes float {
        0%, 100% { transform: translate(0, 0) rotate(0deg); }
        33% { transform: translate(2%, 2%) rotate(1deg); }
        66% { transform: translate(-1%, 1%) rotate(-1deg); }
    }

    /* ========================================
       SIDEBAR - Sleek & Modern
       ======================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12141c 0%, #1a1d29 100%);
        border-right: 1px solid rgba(255, 142, 83, 0.15);
    }
    
    /* Collapsed sidebar - ensure visibility */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        background: linear-gradient(180deg, #12141c 0%, #1a1d29 100%) !important;
        min-width: 2.5rem !important;
    }
    
    /* Sidebar collapse/expand button */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"] {
        color: #FF6B6B !important;
        background: rgba(255, 107, 107, 0.1) !important;
        border-radius: 8px !important;
    }
    
    button[data-testid="stSidebarCollapseButton"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover {
        background: rgba(255, 107, 107, 0.2) !important;
    }
    
    /* Sidebar expand button when collapsed - FORCE VISIBILITY */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(180deg, #1a1d29 0%, #12141c 100%) !important;
        border-right: 2px solid rgba(255, 107, 107, 0.3) !important;
        min-width: 3rem !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarNavCollapseIcon"],
    button[kind="headerNoPadding"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        color: #FF6B6B !important;
        background: rgba(255, 107, 107, 0.2) !important;
        border-radius: 8px !important;
        min-width: 2rem !important;
        min-height: 2rem !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #FF6B6B !important;
        stroke: #FF6B6B !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar text - high contrast */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] [data-baseweb="radio"] span {
        color: #e8eaed !important;
        font-weight: 500 !important;
    }

    /* ========================================
       TYPOGRAPHY - Clear & Readable
       ======================================== */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown td, .stMarkdown th {
        color: #e8eaed !important;
        line-height: 1.7;
    }
    
    .stMarkdown strong, .stMarkdown b {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
        font-weight: 700;
    }

    /* Labels - bright & clear */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stSlider label, .stCheckbox label, .stRadio label, .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    /* CHECKBOX TEXT - FORCE WHITE */
    .stCheckbox, .stCheckbox span, .stCheckbox label, .stCheckbox label span,
    .stCheckbox p, .stCheckbox div,
    [data-testid="stCheckbox"], [data-testid="stCheckbox"] span,
    [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p,
    [data-baseweb="checkbox"] span, [data-baseweb="checkbox"] + div {
        color: #ffffff !important;
        font-weight: 500 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* RADIO TEXT - FORCE WHITE */
    .stRadio, .stRadio span, .stRadio label, .stRadio label span,
    .stRadio p, .stRadio div,
    [data-testid="stRadio"], [data-testid="stRadio"] span,
    [data-testid="stRadio"] label, [data-testid="stRadio"] p,
    [data-baseweb="radio"] span, [data-baseweb="radio"] + div {
        color: #ffffff !important;
        font-weight: 500 !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Captions */
    .stCaption, [data-testid="stCaption"] {
        color: #d1d5db !important;
        font-size: 0.875rem !important;
    }
    
    /* Expander text */
    .streamlit-expanderHeader, .streamlit-expanderHeader p,
    .streamlit-expanderContent, .streamlit-expanderContent p,
    .streamlit-expanderContent span, .streamlit-expanderContent label,
    [data-testid="stExpander"], [data-testid="stExpander"] p,
    [data-testid="stExpander"] span {
        color: #ffffff !important;
    }

    /* ========================================
       TABS - Elegant Design
       ======================================== */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(26, 29, 41, 0.85);
        border-radius: 16px;
        border: 1px solid rgba(255, 142, 83, 0.12);
        padding: 24px;
        backdrop-filter: blur(12px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(18, 20, 28, 0.9);
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9ca3af !important;
        font-weight: 600;
        padding: 12px 20px;
        transition: all 0.25s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 142, 83, 0.1);
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.35);
    }

    /* ========================================
       BUTTONS - Vibrant & Interactive
       ======================================== */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(255, 107, 107, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(255, 107, 107, 0.45);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #FF8E53 0%, #e85d04 100%);
        box-shadow: 0 4px 16px rgba(255, 142, 83, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        box-shadow: 0 8px 28px rgba(255, 142, 83, 0.45);
    }

    /* ========================================
       INPUTS - Clean & Modern
       ======================================== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > textarea {
        background: rgba(18, 20, 28, 0.9) !important;
        border: 1.5px solid rgba(156, 163, 175, 0.25) !important;
        border-radius: 10px !important;
        color: #f3f4f6 !important;
        font-size: 0.95rem !important;
        padding: 12px 14px !important;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > textarea:focus {
        border-color: #FF8E53 !important;
        box-shadow: 0 0 0 3px rgba(255, 142, 83, 0.15) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Select boxes */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(18, 20, 28, 0.9) !important;
        border: 1.5px solid rgba(156, 163, 175, 0.25) !important;
        border-radius: 10px !important;
        color: #f3f4f6 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #FF8E53 0%, #e85d04 100%) !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #9ca3af !important;
    }

    /* ========================================
       METRICS - Glassmorphic Cards
       ======================================== */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(26, 29, 41, 0.9) 0%, rgba(18, 20, 28, 0.9) 100%);
        border: 1px solid rgba(255, 142, 83, 0.15);
        border-radius: 14px;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(255, 107, 107, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255, 107, 107, 0.15);
    }
    
    [data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    /* ========================================
       CUSTOM CLASSES - Premium Elements
       ======================================== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 40%, #FFE66D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        color: #9ca3af !important;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, rgba(26, 29, 41, 0.95) 0%, rgba(18, 20, 28, 0.95) 100%);
        border: 1px solid rgba(255, 142, 83, 0.12);
        border-radius: 16px;
        padding: 28px;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        border-color: rgba(255, 107, 107, 0.4);
        transform: translateY(-4px);
        box-shadow: 0 16px 32px rgba(255, 107, 107, 0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 16px;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #9ca3af !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e !important;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .badge-warning {
        background: rgba(255, 230, 109, 0.15);
        color: #FFE66D !important;
        border: 1px solid rgba(255, 230, 109, 0.3);
    }
    
    .badge-error {
        background: rgba(255, 107, 107, 0.15);
        color: #FF6B6B !important;
        border: 1px solid rgba(255, 107, 107, 0.3);
    }
    
    .badge-info {
        background: rgba(99, 179, 237, 0.15);
        color: #63B3ED !important;
        border: 1px solid rgba(99, 179, 237, 0.3);
    }

    /* ========================================
       ALERTS & NOTIFICATIONS
       ======================================== */
    .stInfo {
        background: rgba(99, 179, 237, 0.1) !important;
        border: 1px solid rgba(99, 179, 237, 0.3) !important;
        border-radius: 10px !important;
        color: #bee3f8 !important;
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 10px !important;
        color: #bbf7d0 !important;
    }
    
    .stWarning {
        background: rgba(255, 230, 109, 0.1) !important;
        border: 1px solid rgba(255, 230, 109, 0.3) !important;
        border-radius: 10px !important;
        color: #fef3c7 !important;
    }
    
    .stError {
        background: rgba(255, 107, 107, 0.1) !important;
        border: 1px solid rgba(255, 107, 107, 0.3) !important;
        border-radius: 10px !important;
        color: #fed7d7 !important;
    }

    /* ========================================
       CODE & EXPANDERS
       ======================================== */
    .stCodeBlock {
        background: rgba(12, 14, 20, 0.95) !important;
        border: 1px solid rgba(255, 142, 83, 0.15) !important;
        border-radius: 10px !important;
    }
    
    .stCodeBlock code {
        color: #e8eaed !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(18, 20, 28, 0.9) !important;
        border-radius: 10px !important;
        color: #e8eaed !important;
        font-weight: 600 !important;
    }
    
    hr {
        border-color: rgba(255, 142, 83, 0.15) !important;
        margin: 2rem 0 !important;
    }

    /* ========================================
       SCROLLBAR & MISC
       ======================================== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d29;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FF8E53, #e85d04);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* JSON viewer */
    .stJson {
        background: rgba(12, 14, 20, 0.95) !important;
        border-radius: 10px !important;
    }
    
    /* Tooltip */
    [data-testid="stTooltipIcon"] {
        color: #FF8E53 !important;
    }
</style>
"""

# Apply the theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'config': None,
        'training_process': None,
        'training_logs': [],
        'training_running': False,
        'log_queue': None,
        'theme': 'dark',  # 'dark' or 'light'
        # Testing page state
        'test_base_model': None,
        'test_finetuned_model': None,
        'test_tokenizer': None,
        'test_chat_history': [],
        'selected_checkpoint': None,
        'test_models_loaded': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load config if not loaded - prefer current.yaml, fall back to default.yaml
    if st.session_state.config is None:
        current_config_path = PROJECT_ROOT / "configs" / "current.yaml"
        default_config_path = PROJECT_ROOT / "configs" / "default.yaml"
        
        if current_config_path.exists():
            st.session_state.config = Config.from_yaml(str(current_config_path))
        elif default_config_path.exists():
            st.session_state.config = Config.from_yaml(str(default_config_path))
        else:
            st.session_state.config = Config()

init_session_state()


# ============================================================================
# Utility Functions
# ============================================================================

def get_status_badge(condition: bool, success_text: str, fail_text: str) -> str:
    """Generate HTML for status badge."""
    if condition:
        return f'<span class="badge badge-success">✓ {success_text}</span>'
    return f'<span class="badge badge-error">✗ {fail_text}</span>'


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


# ============================================================================
# Page: Home / Dashboard
# ============================================================================

def page_home():
    """Render home/dashboard page."""
    st.markdown('<h1 class="main-title">🚀 MLX LoRA Fine-tuning</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fine-tune LLM models with LoRA on Apple Silicon • Powered by MLX</p>', unsafe_allow_html=True)
    
    config = st.session_state.config
    
    # Status metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hf_token = get_hf_token()
        st.metric(
            label="🔑 HuggingFace Token",
            value="Connected" if hf_token else "Not Set",
            delta="Ready" if hf_token else "Required"
        )
    
    with col2:
        train_exists = Path(config.data.train_file).exists()
        st.metric(
            label="📊 Training Data",
            value="Ready" if train_exists else "Missing",
            delta="OK" if train_exists else "Setup needed"
        )
    
    with col3:
        checkpoints = count_files(Path(config.output.checkpoints_dir))
        st.metric(
            label="💾 Checkpoints",
            value=str(checkpoints),
            delta="saved" if checkpoints else "none yet"
        )
    
    with col4:
        adapters = count_files(Path(config.output.adapters_dir), "*.safetensors")
        st.metric(
            label="🎯 Adapters",
            value=str(adapters),
            delta="trained" if adapters else "none yet"
        )
    
    st.divider()
    
    # Feature cards
    st.markdown('<h2 class="section-header">✨ Features</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Data Preparation</div>
            <div class="feature-desc">Convert your datasets to MLX format with automatic train/validation splits and customizable templates.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🚀</div>
            <div class="feature-title">LoRA Training</div>
            <div class="feature-desc">Fine-tune models efficiently with Low-Rank Adaptation. Full control over hyperparameters and real-time logs.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">☁️</div>
            <div class="feature-title">HuggingFace Hub</div>
            <div class="feature-desc">Seamlessly upload your trained models and checkpoints to the HuggingFace Hub for sharing.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Current configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="section-header">⚙️ Current Configuration</h2>', unsafe_allow_html=True)
        
        config_data = {
            "Model": config.model.name,
            "LoRA Rank": config.lora.rank,
            "LoRA Alpha": config.lora.alpha,
            "Batch Size": config.training.batch_size,
            "Learning Rate": f"{config.training.learning_rate:.0e}",
            "Epochs": config.training.num_epochs,
        }
        
        for key, value in config_data.items():
            st.markdown(f"**{key}:** `{value}`")
    
    with col2:
        st.markdown('<h2 class="section-header">📁 Project Structure</h2>', unsafe_allow_html=True)
        st.code("""
├── data/
│   ├── raw/          # Raw datasets
│   └── processed/    # train.jsonl, valid.jsonl
├── outputs/
│   ├── adapters/     # LoRA weights
│   ├── checkpoints/  # Training checkpoints
│   └── logs/         # Training logs
└── configs/          # YAML configurations
        """, language="text")
    
    # Quick actions
    st.divider()
    st.markdown('<h2 class="section-header">⚡ Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📊 Prepare Data", use_container_width=True):
            st.session_state.page = "data"
            st.rerun()
    with col2:
        if st.button("🚀 Train Model", use_container_width=True):
            st.session_state.page = "train"
            st.rerun()
    with col3:
        if st.button("☁️ Upload to HuggingFace", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()


# ============================================================================
# Page: Data Preparation
# ============================================================================

def page_data_preparation():
    """Render data preparation page with tabs for different data types."""
    st.markdown('<h1 class="main-title">📊 Data Preparation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Convert your content into training data — files, folders, or AI-powered</p>', unsafe_allow_html=True)
    
    # Import additional functions for processing
    from src.data_utils import (
        preprocess_raw_text, 
        clean_text, 
        chunk_text,
        create_train_val_split,
        save_jsonl,
        process_folder,
        load_helper_model,
        preprocess_with_llm,
        # Open Router functions
        is_openrouter_configured,
        get_openrouter_config,
        preprocess_with_openrouter,
        # Two-Agent system
        preprocess_with_agents,
    )
    
    # Three main tabs
    tab1, tab2, tab3 = st.tabs(["📋 Structured JSON", "📝 Raw Text / Folder", "🤖 Agent-Assisted (High Quality)"])
    
    # ========== TAB 1: Structured JSON Data ==========
    with tab1:
        st.markdown('<h2 class="section-header">📋 Convert Structured Dataset</h2>', unsafe_allow_html=True)
        st.markdown("Use this for JSON/JSONL files with instruction-response pairs.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_file = file_picker(
                "📄 Input File",
                default_path="data/raw/dataset.json",
                file_types=[".json", ".jsonl"],
                key="json_input"
            )
            
            output_dir = folder_picker(
                "📁 Output Directory",
                default_path="data/processed",
                key="json_output"
            )
            
            val_split = st.slider(
                "📊 Validation Split (%)",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
                help="Percentage of examples reserved for validation",
                key="json_val"
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                instruction_key = st.text_input(
                    "🔑 Instruction Key",
                    value="instruction",
                    help="JSON key for input/instruction field"
                )
            with col_b:
                response_key = st.text_input(
                    "🔑 Response Key", 
                    value="response",
                    help="JSON key for output/response field"
                )
            
            with st.expander("📝 Custom Template (Optional)"):
                template = st.text_area(
                    "Prompt Template",
                    value="",
                    placeholder="### Instruction:\n{instruction}\n\n### Response:\n{response}",
                    help="Use {instruction} and {response} as placeholders"
                )
        
        with col2:
            st.markdown("### 📋 Preview")
            input_path = Path(input_file)
            if input_path.exists():
                try:
                    data = load_dataset(input_file)
                    st.success(f"✓ Found **{len(data)}** examples")
                    if len(data) > 0:
                        st.json(data[0])
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("⚠️ File not found")
        
        if st.button("🔄 Convert JSON Data", type="primary", use_container_width=True, key="convert_json"):
            input_path = Path(input_file)
            if not input_path.exists():
                st.error("❌ Input file does not exist")
            else:
                with st.spinner("Converting data..."):
                    try:
                        train_path, val_path = convert_to_mlx_format(
                            input_path=input_file,
                            output_dir=output_dir,
                            val_ratio=val_split / 100,
                            instruction_key=instruction_key,
                            response_key=response_key,
                            template=template if template else None,
                        )
                        st.session_state.config.data.train_file = str(train_path)
                        st.session_state.config.data.valid_file = str(val_path)
                        st.success(f"✅ Saved training ({train_path.name}) and validation ({val_path.name}) files!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # ========== TAB 2: Raw Text / Folder Processing ==========
    with tab2:
        st.markdown('<h2 class="section-header">📝 Process Raw Text or Folder</h2>', unsafe_allow_html=True)
        st.markdown("Process text from files, folders, or paste directly — with optional AI assistance.")
        
        # Input source selection
        input_source = st.radio(
            "📂 Input Source",
            options=["📄 Single File", "📁 Folder (Multiple Files)", "📋 Paste Text"],
            horizontal=True,
            help="Choose your content source"
        )
        
        raw_text = ""
        files_to_process = []
        
        # ---- INPUT SECTION ----
        if input_source == "📄 Single File":
            col1, col2 = st.columns([2, 1])
            with col1:
                text_file = file_picker(
                    "📄 Text File",
                    default_path="data/raw/transcript.txt",
                    file_types=[".txt", ".md", ".rst", ".text", ".log"],
                    key="raw_text_file"
                )
                if Path(text_file).exists():
                    try:
                        with open(text_file, "r", encoding="utf-8") as f:
                            raw_text = f.read()
                        st.success(f"✓ Loaded {len(raw_text):,} characters")
                    except Exception as e:
                        st.error(f"Error loading: {e}")
                else:
                    st.warning("⚠️ File not found")
            with col2:
                if raw_text:
                    st.markdown("### Preview")
                    st.text(raw_text[:400] + "..." if len(raw_text) > 400 else raw_text)
        
        elif input_source == "📁 Folder (Multiple Files)":
            col1, col2 = st.columns([2, 1])
            with col1:
                folder_path = folder_picker(
                    "📁 Folder",
                    default_path="data/raw/transcripts",
                    key="raw_folder_path"
                )
                file_exts = st.multiselect(
                    "📄 File Extensions",
                    options=[".txt", ".md", ".rst", ".text", ".log"],
                    default=[".txt", ".md"],
                    help="Which file types to process"
                )
            with col2:
                st.markdown("### Folder Contents")
                if Path(folder_path).exists() and Path(folder_path).is_dir():
                    for ext in file_exts:
                        files_to_process.extend(Path(folder_path).glob(f"*{ext}"))
                        files_to_process.extend(Path(folder_path).glob(f"**/*{ext}"))
                    files_to_process = sorted(set(files_to_process))
                    if files_to_process:
                        st.success(f"✓ Found **{len(files_to_process)}** files")
                        with st.expander("View Files"):
                            for f in files_to_process[:15]:
                                st.text(f"📄 {f.name}")
                            if len(files_to_process) > 15:
                                st.text(f"... and {len(files_to_process) - 15} more")
                    else:
                        st.warning("No matching files")
                else:
                    st.warning("⚠️ Folder not found")
        
        else:  # Paste Text
            raw_text = st.text_area(
                "📋 Paste your text here",
                height=180,
                placeholder="Paste your transcript, book chapter, or article text here...",
                help="Paste the raw text content you want to convert into training data"
            )
            if raw_text:
                st.info(f"📊 {len(raw_text):,} characters entered")
        
        st.divider()
        
        # ---- PROCESSING OPTIONS ----
        st.markdown('<h3 class="section-header">⚙️ Processing Options</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Cleaning**")
            clean_timestamps = st.checkbox("🕐 Remove timestamps", value=True, help="Remove YouTube-style timestamps")
            clean_urls = st.checkbox("🔗 Remove URLs", value=False, help="Remove HTTP/HTTPS links")
            clean_speakers = st.checkbox("👤 Remove speaker labels", value=False, help="Remove 'Speaker 1:' patterns")
        
        with col2:
            st.markdown("**Chunking**")
            chunk_size = st.slider("📏 Chunk Size", min_value=200, max_value=4000, value=1000, step=100, help="Target chunk size in characters")
            chunk_overlap = st.slider("🔄 Overlap", min_value=0, max_value=500, value=100, step=50, help="Overlap between chunks")
        
        with col3:
            st.markdown("**Output Format**")
            output_format = st.selectbox(
                "📤 Format",
                options=["completion", "qa", "knowledge", "raw"],
                format_func=lambda x: {"completion": "📝 Completion", "qa": "❓ Q&A", "knowledge": "🧠 Knowledge", "raw": "📄 Raw"}[x],
                help="How to format training data"
            )
            if output_format == "knowledge":
                topic = st.text_input("📌 Topic", value="this topic", help="Topic for knowledge format")
            else:
                topic = "the content"
        
        # ---- AI ENHANCEMENT ----
        with st.expander("🤖 AI-Enhanced Generation (Optional)", expanded=False):
            st.markdown("Use AI to generate intelligent Q&A pairs or summaries. Choose between fast cloud API or local model.")
            
            use_ai = st.checkbox("✨ Enable AI Enhancement", value=False, help="Use AI for smarter data generation")
            
            if use_ai:
                # Check if Open Router is configured
                openrouter_ready = is_openrouter_configured()
                
                # AI Provider selection
                ai_provider = st.radio(
                    "🔌 AI Provider",
                    options=["openrouter", "local"],
                    format_func=lambda x: {
                        "openrouter": "☁️ Open Router (Fast Cloud API)",
                        "local": "💻 Local Model (Slower, No API needed)"
                    }[x],
                    horizontal=True,
                    help="Open Router is faster but requires an API key. Local runs on your machine."
                )
                
                if ai_provider == "openrouter":
                    # Open Router configuration
                    if openrouter_ready:
                        config = get_openrouter_config()
                        st.success(f"✅ Open Router configured with model: `{config['model']}`")
                        
                        # Allow custom model override
                        col_a, col_b = st.columns([2, 1])
                        with col_a:
                            openrouter_model = st.text_input(
                                "🤖 Model", 
                                value=config["model"], 
                                help="Open Router model ID (e.g., qwen/qwen3-0.6b-04-28, openai/gpt-4o-mini)",
                                key="openrouter_model_input"
                            )
                        with col_b:
                            st.markdown("#### ")
                            st.markdown("[📖 View Models](https://openrouter.ai/models)")
                        
                        st.session_state.openrouter_model = openrouter_model
                        st.session_state.use_openrouter = True
                    else:
                        st.warning("⚠️ Open Router not configured. Add `OPENROUTER_API_KEY` to your `.env` file.")
                        st.markdown("""
                        **To configure Open Router:**
                        1. Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys)
                        2. Add to your `.env` file:
                        ```
                        OPENROUTER_API_KEY=sk-or-your-key-here
                        OPENROUTER_MODEL=qwen/qwen3-0.6b-04-28
                        ```
                        3. Restart the app
                        """)
                        st.session_state.use_openrouter = False
                else:
                    # Local model configuration
                    st.session_state.use_openrouter = False
                    
                    if 'helper_model' not in st.session_state:
                        st.session_state.helper_model = None
                        st.session_state.helper_tokenizer = None
                    
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        helper_model_name = st.text_input("🤖 Model", value="Qwen/Qwen3-0.6B", help="Small model for data generation")
                    with col_b:
                        if st.button("📥 Load Model"):
                            with st.spinner(f"Loading {helper_model_name}..."):
                                try:
                                    model, tokenizer = load_helper_model(helper_model_name)
                                    st.session_state.helper_model = model
                                    st.session_state.helper_tokenizer = tokenizer
                                    st.success("✅ Model loaded!")
                                except Exception as e:
                                    st.error(f"❌ {e}")
                    
                    if st.session_state.helper_model:
                        st.success("✅ AI model ready")
                    else:
                        st.warning("⚠️ Load AI model first to enable")
                
                # AI format selection (common to both providers)
                ai_format = st.radio(
                    "AI Generation", 
                    ["qa", "summary"], 
                    format_func=lambda x: {"qa": "❓ Q&A", "summary": "📝 Summary"}[x], 
                    horizontal=True
                )
            else:
                ai_format = "qa"
                st.session_state.use_openrouter = False
        
        st.divider()
        
        # ---- OUTPUT & GENERATE ----
        col1, col2 = st.columns([1, 1])
        
        with col1:
            raw_output_dir = st.text_input("📁 Output Directory", value="data/processed", key="raw_output", help="Where to save training files")
            raw_val_split = st.slider("📊 Validation Split (%)", min_value=5, max_value=30, value=10, step=5, key="raw_val")
        
        with col2:
            st.markdown("### 🚀 Generate")
            
            has_content = bool(raw_text) or bool(files_to_process)
            
            if st.button("✨ Generate Training Data", type="primary", use_container_width=True, disabled=not has_content):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    all_examples = []
                    
                    # Determine AI mode
                    use_openrouter = use_ai and st.session_state.get('use_openrouter', False)
                    use_local_model = use_ai and not use_openrouter and st.session_state.get('helper_model') is not None
                    
                    # Process folder if applicable
                    if files_to_process:
                        for i, file_path in enumerate(files_to_process):
                            progress_bar.progress((i + 1) / len(files_to_process))
                            status_text.text(f"Processing: {file_path.name}")
                            
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                            
                            if use_openrouter:
                                # Use Open Router API
                                examples = preprocess_with_openrouter(
                                    text, 
                                    output_format=ai_format, 
                                    chunk_size=chunk_size,
                                    clean_timestamps=clean_timestamps, 
                                    clean_urls=clean_urls,
                                    model=st.session_state.get('openrouter_model'),
                                )
                            elif use_local_model:
                                # Use local model
                                examples = preprocess_with_llm(
                                    text, st.session_state.helper_model, st.session_state.helper_tokenizer,
                                    output_format=ai_format, chunk_size=chunk_size,
                                    clean_timestamps=clean_timestamps, clean_urls=clean_urls,
                                )
                            else:
                                examples = preprocess_raw_text(
                                    text, output_format=output_format, chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap, clean_timestamps=clean_timestamps,
                                    clean_urls=clean_urls, topic=topic,
                                )
                            all_examples.extend(examples)
                    else:
                        # Process single text
                        def progress_cb(current, total):
                            # Clamp progress to max 1.0 to avoid Streamlit error
                            progress_value = min((current + 1) / max(total, 1), 1.0)
                            progress_bar.progress(progress_value)
                            status_text.text(f"Processing chunk {min(current + 1, total)}/{total}...")
                        
                        if use_openrouter:
                            # Use Open Router API
                            all_examples = preprocess_with_openrouter(
                                raw_text, 
                                output_format=ai_format, 
                                chunk_size=chunk_size,
                                clean_timestamps=clean_timestamps, 
                                clean_urls=clean_urls,
                                model=st.session_state.get('openrouter_model'),
                                progress_callback=progress_cb,
                            )
                        elif use_local_model:
                            # Use local model
                            all_examples = preprocess_with_llm(
                                raw_text, st.session_state.helper_model, st.session_state.helper_tokenizer,
                                output_format=ai_format, chunk_size=chunk_size,
                                clean_timestamps=clean_timestamps, clean_urls=clean_urls,
                                progress_callback=progress_cb,
                            )
                        else:
                            all_examples = preprocess_raw_text(
                                raw_text, output_format=output_format, chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap, clean_timestamps=clean_timestamps,
                                clean_urls=clean_urls, topic=topic,
                            )
                    
                    if not all_examples:
                        st.error("❌ No examples generated. Check your content.")
                    else:
                        train_data, val_data = create_train_val_split(all_examples, val_ratio=raw_val_split / 100)
                        
                        output_path = Path(raw_output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        train_path = output_path / "train.jsonl"
                        val_path = output_path / "valid.jsonl"
                        
                        save_jsonl(train_data, train_path)
                        save_jsonl(val_data, val_path)
                        
                        st.session_state.config.data.train_file = str(train_path)
                        st.session_state.config.data.valid_file = str(val_path)
                        
                        status_text.empty()
                        st.success(f"✅ Generated **{len(train_data)}** training + **{len(val_data)}** validation examples!")
                        
                        with st.expander("View Sample"):
                            st.code(all_examples[0]["text"][:600], language="text")
                            
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # ========== TAB 3: Agent-Assisted High Quality Generation ==========
    with tab3:
        st.markdown('<h2 class="section-header">🤖 Agent-Assisted Data Generation</h2>', unsafe_allow_html=True)
        st.markdown("""
        Generate **high-quality** Q&A training data using a two-agent system:
        1. **Meta-Agent**: Analyzes your fine-tuning intention and creates a specialized prompt
        2. **Generator-Agent**: Uses that prompt to create focused, actionable Q&A pairs
        """)
        
        # Check if Open Router is configured
        openrouter_ready = is_openrouter_configured()
        
        if not openrouter_ready:
            st.error("⚠️ **Open Router API required**. Add `OPENROUTER_API_KEY` to your `.env` file.")
            st.markdown("""
            **To configure:**
            1. Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys)
            2. Add to `.env`: `OPENROUTER_API_KEY=sk-or-your-key-here`
            3. Restart the app
            """)
        else:
            st.success("✅ Open Router configured and ready")
            
            st.markdown("---")
            st.markdown('<h3 class="section-header">🎯 Fine-Tuning Objective</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                intention = st.text_area(
                    "📌 What do you want the model to learn?",
                    value="I want to train a model that gives practical business and sales advice, focusing on lead generation, customer acquisition, and revenue growth strategies.",
                    height=120,
                    help="Describe your fine-tuning goal. Be specific about the knowledge and skills you want the model to acquire.",
                    key="agent_intention"
                )
                
                personality = st.text_area(
                    "🎭 What personality/style should the model have?",
                    value="Direct and no-nonsense. Uses concrete numbers and examples. Gives actionable advice rather than vague suggestions. Speaks like a successful entrepreneur sharing hard-won lessons.",
                    height=100,
                    help="Describe how the model should respond - its tone, style, and approach.",
                    key="agent_personality"
                )
            
            with col2:
                st.markdown("**📋 Types of Questions to Generate**")
                q_practical = st.checkbox("💡 Practical / How-to", value=True, help="Step-by-step actionable questions")
                q_strategic = st.checkbox("🎯 Strategic / Decision-making", value=True, help="Why and when to apply strategies")
                q_application = st.checkbox("🔧 Application / Implementation", value=True, help="How to apply concepts in real scenarios")
                q_mistakes = st.checkbox("⚠️ Mistakes / What to avoid", value=True, help="Common pitfalls and how to avoid them")
                q_comparison = st.checkbox("⚖️ Comparisons / Trade-offs", value=False, help="Comparing approaches and their trade-offs")
                
                question_types = []
                if q_practical: question_types.append("Practical/How-to")
                if q_strategic: question_types.append("Strategic/Decision-making")
                if q_application: question_types.append("Application/Implementation")
                if q_mistakes: question_types.append("Mistakes/What to avoid")
                if q_comparison: question_types.append("Comparisons/Trade-offs")
            
            st.markdown("---")
            st.markdown('<h3 class="section-header">📄 Source Content</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                agent_input_file = file_picker(
                    "📄 Text File",
                    default_path="data/raw/raw_txt.txt",
                    file_types=[".txt", ".md"],
                    key="agent_input_file"
                )
                
                agent_output_dir = folder_picker(
                    "📁 Output Directory",
                    default_path="data/processed",
                    key="agent_output_dir"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    agent_chunk_size = st.slider("📏 Chunk Size", min_value=500, max_value=3000, value=1500, step=100, help="Larger = more context per chunk", key="agent_chunk")
                    agent_questions = st.slider("❓ Q&A per Chunk", min_value=1, max_value=5, value=2, step=1, help="Number of Q&A pairs to generate per text chunk", key="agent_qa_count")
                with col_b:
                    agent_val_split = st.slider("📊 Validation Split (%)", min_value=5, max_value=30, value=10, step=5, key="agent_val")
            
            with col2:
                st.markdown("### Source Preview")
                if Path(agent_input_file).exists():
                    with open(agent_input_file, "r", encoding="utf-8") as f:
                        source_text = f.read()
                    st.success(f"✓ {len(source_text):,} characters")
                    st.text(source_text[:300] + "..." if len(source_text) > 300 else source_text)
                else:
                    st.warning("⚠️ File not found")
                    source_text = ""
            
            st.markdown("---")
            
            # Model Configuration expander
            with st.expander("🔧 Model Configuration", expanded=True):
                st.markdown("**Choose how to run the AI models:**")
                
                agent_provider = st.radio(
                    "🔌 Provider",
                    options=["openrouter", "local"],
                    format_func=lambda x: {
                        "openrouter": "☁️ OpenRouter API (Fast, requires API key)",
                        "local": "💻 Local HuggingFace (Slower, no API needed)"
                    }[x],
                    horizontal=True,
                    key="agent_provider"
                )
                
                if agent_provider == "openrouter":
                    col1, col2 = st.columns(2)
                    with col1:
                        meta_model = st.text_input(
                            "🧠 Meta-Agent Model",
                            value="mistralai/devstral-2512:free",
                            help="Model for analyzing intention (smarter is better)",
                            key="meta_model"
                        )
                    with col2:
                        gen_model = st.text_input(
                            "⚡ Generator Model",
                            value=get_openrouter_config()["model"],
                            help="Model for Q&A generation (can be faster/cheaper)",
                            key="gen_model"
                        )
                    st.markdown("[📖 Browse OpenRouter Models](https://openrouter.ai/models)")
                    
                    # Store provider choice
                    use_local_hf = False
                    local_model_name = None
                    
                else:  # Local HuggingFace
                    st.info("💡 Local mode uses HuggingFace models on your machine. Requires more RAM and is slower.")
                    
                    local_model_name = st.text_input(
                        "🤖 HuggingFace Model",
                        value="Qwen/Qwen3-0.6B",
                        help="Model ID from HuggingFace Hub",
                        key="local_hf_model"
                    )
                    
                    # Check if model is loaded
                    if 'agent_local_model' not in st.session_state:
                        st.session_state.agent_local_model = None
                        st.session_state.agent_local_tokenizer = None
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if st.button("📥 Load Model", key="load_local_agent_model"):
                            with st.spinner(f"Loading {local_model_name}..."):
                                try:
                                    model, tokenizer = load_helper_model(local_model_name)
                                    st.session_state.agent_local_model = model
                                    st.session_state.agent_local_tokenizer = tokenizer
                                    st.success("✅ Model loaded!")
                                except Exception as e:
                                    st.error(f"❌ Failed: {e}")
                    with col2:
                        if st.session_state.agent_local_model:
                            st.success(f"✅ Model ready: {local_model_name}")
                        else:
                            st.warning("⚠️ Load model before generating")
                    
                    # For local mode, we'll use same model for both agents
                    meta_model = None
                    gen_model = None
                    use_local_hf = True
            
            # Generate button
            can_generate = source_text and (not use_local_hf or st.session_state.get('agent_local_model'))
            
            if st.button("🚀 Generate High-Quality Dataset", type="primary", use_container_width=True, disabled=not can_generate):
                progress_bar = st.progress(0)
                status_text = st.empty()
                prompt_display = st.empty()
                
                try:
                    def progress_cb(current, total):
                        progress_value = min((current + 1) / max(total, 1), 1.0)
                        progress_bar.progress(progress_value)
                        status_text.text(f"⚡ Processing chunk {min(current + 1, total)}/{total}...")
                    
                    if use_local_hf:
                        # Local HuggingFace mode
                        status_text.text("🧠 Processing with local model...")
                        
                        # For local mode, we use the simpler preprocess_with_llm
                        examples = preprocess_with_llm(
                            text=source_text,
                            model=st.session_state.agent_local_model,
                            tokenizer=st.session_state.agent_local_tokenizer,
                            output_format="qa",
                            chunk_size=agent_chunk_size,
                            questions_per_chunk=agent_questions,
                            progress_callback=progress_cb,
                        )
                        specialized_prompt = f"[Local Mode] Using {local_model_name} to generate Q&A pairs based on intention: {intention}"
                    else:
                        # OpenRouter mode with two-agent system
                        status_text.text("🧠 Meta-Agent analyzing content and generating specialized prompt...")
                        
                        examples, specialized_prompt = preprocess_with_agents(
                            text=source_text,
                            intention=intention,
                            personality=personality,
                            question_types=question_types,
                            chunk_size=agent_chunk_size,
                            questions_per_chunk=agent_questions,
                            meta_model=meta_model,
                            generator_model=gen_model,
                            progress_callback=progress_cb,
                        )
                    
                    if not examples:
                        st.error("❌ No examples generated. Check your content or try different settings.")
                    else:
                        # Save results
                        train_data, val_data = create_train_val_split(examples, val_ratio=agent_val_split / 100)
                        
                        output_path = Path(agent_output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        
                        train_path = output_path / "train.jsonl"
                        val_path = output_path / "valid.jsonl"
                        
                        save_jsonl(train_data, train_path)
                        save_jsonl(val_data, val_path)
                        
                        st.session_state.config.data.train_file = str(train_path)
                        st.session_state.config.data.valid_file = str(val_path)
                        
                        status_text.empty()
                        st.success(f"✅ Generated **{len(train_data)}** training + **{len(val_data)}** validation examples!")
                        
                        # Show the specialized prompt that was used
                        with st.expander("📋 Specialized Prompt Used (from Meta-Agent)"):
                            st.code(specialized_prompt, language="text")
                        
                        # Show sample output
                        with st.expander("🔍 Sample Generated Q&A"):
                            for i, example in enumerate(examples[:3]):
                                st.markdown(f"**Example {i+1}:**")
                                st.code(example["text"], language="text")
                                st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
    
    # Footer
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📂 View Processed Files", use_container_width=True):
            output_path = Path(st.session_state.config.data.train_file).parent
            if output_path.exists():
                files = list(output_path.glob("*.jsonl"))
                if files:
                    for f in files:
                        try:
                            data = load_dataset(f)
                            st.info(f"📄 `{f.name}`: **{len(data)}** examples")
                        except:
                            st.warning(f"Could not read {f.name}")
                else:
                    st.warning("No JSONL files found")
            else:
                st.warning("Output directory does not exist")
    with col2:
        if st.button("🚀 Go to Training", use_container_width=True):
            st.session_state.page = "train"
            st.rerun()


# ============================================================================
# Page: Training
# ============================================================================

# Preset configurations for different use cases
TRAINING_PRESETS = {
    "🚀 Quick Test": {
        "description": "Fast training to test your setup. Low quality but quick results.",
        "lora_rank": 4,
        "lora_alpha": 8,
        "batch_size": 2,
        "epochs": 1,
        "learning_rate": 2e-4,
    },
    "⚖️ Balanced": {
        "description": "Good balance between training time and quality. Recommended for most users.",
        "lora_rank": 8,
        "lora_alpha": 16,
        "batch_size": 4,
        "epochs": 3,
        "learning_rate": 1e-4,
    },
    "🎯 High Quality": {
        "description": "Longer training for better results. Use when you have more time.",
        "lora_rank": 16,
        "lora_alpha": 32,
        "batch_size": 4,
        "epochs": 5,
        "learning_rate": 5e-5,
    },
    "🔬 Maximum Quality": {
        "description": "Maximum training capacity. Best for final production models.",
        "lora_rank": 32,
        "lora_alpha": 64,
        "batch_size": 2,
        "epochs": 10,
        "learning_rate": 2e-5,
    },
}

# Popular model suggestions
POPULAR_MODELS = [
    ("meta-llama/Llama-3.2-1B", "Llama 3.2 1B - Fast, good for testing"),
    ("meta-llama/Llama-3.2-3B", "Llama 3.2 3B - Balanced performance"),
    ("Qwen/Qwen2.5-1.5B", "Qwen 2.5 1.5B - Excellent multilingual"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini - Very efficient"),
    ("google/gemma-2-2b", "Gemma 2 2B - Google's compact model"),
]


def page_training():
    """Render training configuration and execution page."""
    st.markdown('<h1 class="main-title">🚀 Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fine-tune your model with just a few clicks — or dive deep into advanced settings</p>', unsafe_allow_html=True)
    
    config = st.session_state.config
    
    # Initialize advanced mode in session state
    if 'advanced_mode' not in st.session_state:
        st.session_state.advanced_mode = False
    
    # Mode toggle
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        advanced_mode = st.toggle(
            "⚙️ Advanced Mode",
            value=st.session_state.advanced_mode,
            help="Enable to access all configuration options"
        )
        st.session_state.advanced_mode = advanced_mode
    
    st.divider()
    
    if not advanced_mode:
        # ========== SIMPLE MODE ==========
        render_simple_mode(config)
    else:
        # ========== ADVANCED MODE ==========
        render_advanced_mode(config)
    
    st.divider()
    
    # ========== EXECUTE SECTION (Common to both modes) ==========
    render_execute_section(config)


def render_simple_mode(config):
    """Render the simplified beginner-friendly interface."""
    
    # Step 1: Choose a preset
    st.markdown('<h2 class="section-header">1️⃣ Choose Training Preset</h2>', unsafe_allow_html=True)
    st.markdown("Select a preset based on how much time you have and the quality you need:")
    
    cols = st.columns(len(TRAINING_PRESETS))
    
    selected_preset = None
    for i, (preset_name, preset_config) in enumerate(TRAINING_PRESETS.items()):
        with cols[i]:
            # Create a card-like button
            if st.button(preset_name, key=f"preset_{i}", use_container_width=True):
                selected_preset = preset_name
                # Apply preset
                config.lora.rank = preset_config["lora_rank"]
                config.lora.alpha = preset_config["lora_alpha"]
                config.training.batch_size = preset_config["batch_size"]
                config.training.num_epochs = preset_config["epochs"]
                config.training.learning_rate = preset_config["learning_rate"]
                st.success(f"✅ Applied: {preset_name}")
            st.caption(preset_config["description"])
    
    st.info(f"""
    **Current Settings:** LoRA rank={config.lora.rank}, epochs={config.training.num_epochs}, 
    batch size={config.training.batch_size}, learning rate={config.training.learning_rate:.0e}
    """)
    
    st.divider()
    
    # Step 2: Select Model
    st.markdown('<h2 class="section-header">2️⃣ Select Base Model</h2>', unsafe_allow_html=True)
    
    config.model.name = st.text_input(
        "🤖 HuggingFace Model ID",
        value=config.model.name,
        help="Enter the HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B, Qwen/Qwen2.5-1.5B, microsoft/Phi-3-mini-4k-instruct). The model will be automatically downloaded when training starts.",
        placeholder="meta-llama/Llama-3.2-1B"
    )
    
    # Suggestions
    with st.expander("💡 Recommended Models for Apple Silicon"):
        st.markdown("""
        | Model | Size | RAM | Best For |
        |-------|------|-----|----------|
        | `HuggingFaceTB/SmolLM2-135M` | 135M | ~1GB | Ultra-fast research, testing |
        | `HuggingFaceTB/SmolLM2-360M` | 360M | ~2GB | Efficient on-device tasks |
        | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~2GB | High instruction-following |
        | `h2oai/h2o-danube3-500m-base`| 500M | ~2GB | Balanced SLM experimentation |
        | `meta-llama/Llama-3.2-1B` | 1B | 4GB | Fast testing, quick iterations |
        | `meta-llama/Llama-3.2-3B` | 3B | 8GB | Balanced quality and speed |
        | `Qwen/Qwen2.5-7B` | 7B | 16GB+ | High quality production |
        """)
    
    st.divider()
    
    # Step 3: Training Data
    st.markdown('<h2 class="section-header">3️⃣ Training Data</h2>', unsafe_allow_html=True)
    
    train_exists = Path(config.data.train_file).exists()
    valid_exists = Path(config.data.valid_file).exists()
    
    if train_exists:
        try:
            train_data = load_dataset(config.data.train_file)
            st.success(f"✅ Training data ready: **{len(train_data)} examples** from `{config.data.train_file}`")
        except:
            st.warning("⚠️ Could not load training data")
    else:
        st.warning("⚠️ No training data found. Go to **Prepare Data** page first.")
        if st.button("📊 Go to Data Preparation"):
            st.session_state.page = "data"
            st.rerun()
    
    # Optional: Quick settings adjustments
    with st.expander("🔧 Quick Adjustments (Optional)"):
        col1, col2 = st.columns(2)
        with col1:
            config.training.num_epochs = st.slider(
                "🔄 Training Epochs",
                min_value=1,
                max_value=10,
                value=config.training.num_epochs,
                help="More epochs = longer training but potentially better results"
            )
        with col2:
            config.training.batch_size = st.slider(
                "📦 Batch Size",
                min_value=1,
                max_value=8,
                value=min(config.training.batch_size, 8),
                help="Higher = faster but uses more memory"
            )


def render_advanced_mode(config):
    """Render the advanced interface with full control."""
    
    st.info("🔬 **Advanced Mode** — Full control over all training parameters")
    
    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Model & LoRA", "📈 Training", "⚙️ Advanced"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Base Model")
            config.model.name = st.text_input(
                "🏷️ Model Name/Path",
                value=config.model.name,
                help="HuggingFace model ID or local path"
            )
            
            config.model.max_seq_length = st.slider(
                "📏 Max Sequence Length",
                min_value=256,
                max_value=8192,
                value=config.model.max_seq_length,
                step=256,
                help="Longer sequences need more memory"
            )
        
        with col2:
            st.markdown("### LoRA Configuration")
            config.lora.rank = st.slider(
                "📊 Rank (r)",
                min_value=2,
                max_value=128,
                value=config.lora.rank,
                help="Higher = more capacity, more memory"
            )
            
            config.lora.alpha = st.slider(
                "⚖️ Alpha",
                min_value=4,
                max_value=256,
                value=config.lora.alpha,
                help=f"Effective scale: {config.lora.alpha/config.lora.rank:.2f}x"
            )
            
            config.lora.dropout = st.slider(
                "💧 Dropout",
                min_value=0.0,
                max_value=0.5,
                value=config.lora.dropout,
                step=0.01,
                help="Regularization to prevent overfitting"
            )
        
        st.markdown("### Target Modules")
        all_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        config.lora.target_modules = st.multiselect(
            "Select layers to apply LoRA",
            options=all_modules,
            default=config.lora.target_modules,
            help="More modules = more trainable parameters"
        )
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Core")
            config.training.learning_rate = st.number_input(
                "📉 Learning Rate",
                min_value=1e-7,
                max_value=1e-2,
                value=config.training.learning_rate,
                format="%.2e",
                step=1e-5
            )
            
            config.training.batch_size = st.number_input(
                "📦 Batch Size",
                min_value=1,
                max_value=32,
                value=config.training.batch_size
            )
            
            config.training.num_epochs = st.number_input(
                "🔄 Epochs",
                min_value=1,
                max_value=100,
                value=config.training.num_epochs
            )
        
        with col2:
            st.markdown("### Optimization")
            config.training.warmup_steps = st.number_input(
                "🔥 Warmup Steps",
                min_value=0,
                max_value=5000,
                value=config.training.warmup_steps
            )
            
            config.training.weight_decay = st.number_input(
                "⚖️ Weight Decay",
                min_value=0.0,
                max_value=0.5,
                value=config.training.weight_decay,
                format="%.4f"
            )
            
            config.training.max_grad_norm = st.number_input(
                "📏 Gradient Clipping",
                min_value=0.1,
                max_value=10.0,
                value=config.training.max_grad_norm,
                format="%.2f"
            )
        
        with col3:
            st.markdown("### Checkpointing")
            config.training.save_steps = st.number_input(
                "💾 Save Steps",
                min_value=10,
                max_value=10000,
                value=config.training.save_steps
            )
            
            config.training.eval_steps = st.number_input(
                "📊 Eval Steps",
                min_value=10,
                max_value=5000,
                value=config.training.eval_steps
            )
            
            config.training.logging_steps = st.number_input(
                "📝 Log Steps",
                min_value=1,
                max_value=500,
                value=config.training.logging_steps
            )
    
    with tab3:
        st.markdown("### Data Paths")
        col1, col2 = st.columns(2)
        with col1:
            config.data.train_file = st.text_input(
                "📄 Training File",
                value=config.data.train_file
            )
        with col2:
            config.data.valid_file = st.text_input(
                "📄 Validation File",
                value=config.data.valid_file
            )
        
        st.markdown("### Output Directories")
        col1, col2, col3 = st.columns(3)
        with col1:
            config.output.adapters_dir = st.text_input(
                "📁 Adapters Dir",
                value=config.output.adapters_dir
            )
        with col2:
            config.output.checkpoints_dir = st.text_input(
                "📁 Checkpoints Dir",
                value=config.output.checkpoints_dir
            )
        with col3:
            config.output.logs_dir = st.text_input(
                "📁 Logs Dir",
                value=config.output.logs_dir
            )
        
        st.markdown("### Config Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load Config from YAML", use_container_width=True):
                try:
                    st.session_state.config = Config.from_yaml(str(PROJECT_ROOT / "configs" / "default.yaml"))
                    st.success("✅ Loaded default.yaml")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("💾 Save Config to YAML", use_container_width=True):
                config_path = PROJECT_ROOT / "configs" / "current.yaml"
                config.to_yaml(str(config_path))
                st.success(f"✅ Saved to {config_path}")


def render_execute_section(config):
    """Render the training execution section."""
    st.markdown('<h2 class="section-header">▶️ Start Training</h2>', unsafe_allow_html=True)
    
    # Pre-flight checks
    train_exists = Path(config.data.train_file).exists()
    
    if not train_exists:
        st.error("❌ Training data not found. Please prepare your data first.")
        return
    
    # Summary before training
    with st.expander("📋 Training Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            **Model:** `{config.model.name.split('/')[-1]}`  
            **Sequence Length:** {config.model.max_seq_length}
            """)
        with col2:
            st.markdown(f"""
            **LoRA Rank:** {config.lora.rank} (α={config.lora.alpha})  
            **Dropout:** {config.lora.dropout}
            """)
        with col3:
            st.markdown(f"""
            **Epochs:** {config.training.num_epochs}  
            **Batch Size:** {config.training.batch_size}  
            **Learning Rate:** {config.training.learning_rate:.0e}
            """)
    
    # Terminal command helper
    with st.expander("💻 Run from Terminal (Recommended for Performance)", expanded=False):
        st.info("💡 Running training from the terminal uses significantly fewer resources and prevents the browser from freezing.")
        
        st.markdown("⚠️ **Important:** Ensure you are in the project root directory:")
        st.code(f"cd {PROJECT_ROOT}", language="bash")
        
        st.markdown("Then, run this command to start training with your current settings:")
        st.code("make train-current", language="bash")
        
        with st.expander("Alternative: Direct Python Command"):
            st.code(f"./.venv/bin/python scripts/train.py --config configs/current.yaml", language="bash")
        
        st.markdown(f"""
        **Recommended Workflow:**
        1. Click **'🚀 START TRAINING'** above (this saves your settings to `configs/current.yaml`).
        2. Once logs start, click **'🛑 STOP'** to free up the GPU.
        3. Run `make train-current` in your terminal.
        4. **Close this browser tab**!
        """)

    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        btn_label = "🚀 START TRAINING" if not st.session_state.training_running else "⏳ Training in Progress..."
        if st.button(
            btn_label,
            type="primary",
            use_container_width=True,
            disabled=st.session_state.training_running
        ):
            # Save current config
            config_path = PROJECT_ROOT / "configs" / "current.yaml"
            config.to_yaml(str(config_path))
            
            # Start training subprocess
            st.session_state.training_running = True
            st.session_state.training_logs = []
            
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train.py"),
                "--config", str(config_path)
            ]
            
            st.session_state.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT)
            )
            st.rerun()
    
    with col2:
        if st.button(
            "🛑 STOP",
            use_container_width=True,
            disabled=not st.session_state.training_running
        ):
            if st.session_state.training_process:
                st.session_state.training_process.terminate()
                st.session_state.training_running = False
                st.warning("⚠️ Training stopped")
                st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.training_logs = []
            st.rerun()
    
    # Training logs
    st.markdown("### 📋 Training Logs")
    log_container = st.empty()
    
    
    if st.session_state.training_running and st.session_state.training_process:
        process = st.session_state.training_process
        return_code = process.poll()
        
        # Read available logs
        try:
            # Make output non-blocking
            import os
            try:
                os.set_blocking(process.stdout.fileno(), False)
            except:
                pass
                
            # Read output
            output = process.stdout.read()
            if output:
                new_lines = [line for line in output.split('\n') if line.strip()]
                st.session_state.training_logs.extend(new_lines)
                # Keep logs manageable
                if len(st.session_state.training_logs) > 500:
                    st.session_state.training_logs = st.session_state.training_logs[-500:]
        except Exception as e:
            # Ignore read errors
            pass
            
        # Check if process finished
        if return_code is not None:
            # Use strict type check to avoid false positives with 0
            if return_code == 0:
                st.success("✅ Training completed successfully!")
                st.balloons()
            else:
                st.error(f"❌ Training failed with exit code {return_code}")
                # Try to read stderr if failed
                try:
                    remaining_out = process.stdout.read()
                    if remaining_out:
                         st.session_state.training_logs.extend(remaining_out.split('\n'))
                except:
                    pass
            
            st.session_state.training_running = False
            # Final output update
        else:
            # Rerun quickly to stream logs
            time.sleep(0.1)
            st.rerun()
    
    if st.session_state.training_logs:
        log_container.code('\n'.join(st.session_state.training_logs[-50:]), language="text")
    else:
        log_container.info("💡 Training logs will appear here when you start training...")


# ============================================================================
# Page: Model Testing
# ============================================================================

def get_available_checkpoints():
    """Get list of available checkpoints from outputs/checkpoints folder."""
    checkpoints_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    
    checkpoints = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and (item / "adapters.safetensors").exists():
            checkpoints.append(item.name)
    
    # Sort by name (step-X will be sorted numerically if possible)
    def sort_key(name):
        if name.startswith("step-"):
            try:
                return (1, int(name.split("-")[1]))
            except:
                return (1, 0)
        elif name == "best":
            return (0, 0)
        elif name == "final":
            return (2, 0)
        return (3, name)
    
    return sorted(checkpoints, key=sort_key)


def load_test_models(checkpoint_name: str):
    """Load base and fine-tuned models for comparison."""
    from src.model_utils import load_base_model, apply_lora
    import mlx.core as mx
    
    config = st.session_state.config
    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / checkpoint_name
    adapter_file = checkpoint_path / "adapters.safetensors"
    
    # First, load the adapter weights to extract target modules and rank
    adapters = mx.load(str(adapter_file))
    
    # Extract unique target modules from adapter keys
    # Keys are like: model.layers.0.conv.in_proj.lora_a, model.layers.0.feed_forward.w1.lora_b
    target_modules = set()
    lora_rank = None
    
    for key, value in adapters.items():
        # Extract rank from lora_a weights (shape is [in_features, rank] in MLX)
        if key.endswith('.lora_a') and lora_rank is None:
            lora_rank = value.shape[1]
        
        # Remove the lora_a/lora_b suffix and the model.layers.X prefix
        parts = key.split('.')
        # Find the module name after layers.X
        for i, part in enumerate(parts):
            if part == 'layers' and i + 1 < len(parts):
                # Skip the layer number, get the rest until lora_a/lora_b
                module_parts = parts[i+2:-1]  # Skip 'layers', layer_num, and 'lora_a/b'
                if module_parts:
                    module_name = '.'.join(module_parts)
                    target_modules.add(module_name)
                break
    
    target_modules_list = list(target_modules) if target_modules else None
    lora_rank = lora_rank or config.lora.rank
    lora_alpha = config.lora.alpha if config.lora.alpha else lora_rank * 2
    
    print(f"Extracted from checkpoint: rank={lora_rank}, modules={target_modules_list}")
    
    # Load base model for fine-tuned version
    model, tokenizer = load_base_model(config.model.name)
    st.session_state.test_tokenizer = tokenizer
    
    # Create a copy for base model (without adapters)
    base_model, _ = load_base_model(config.model.name)
    st.session_state.test_base_model = base_model
    
    # Apply LoRA using the extracted parameters from the checkpoint
    model = apply_lora(
        model,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=0.0,  # Dropout not needed for inference
        target_modules=target_modules_list,
    )
    
    # Load adapter weights with strict=False to handle any mismatches
    model.load_weights(list(adapters.items()), strict=False)
    
    st.session_state.test_finetuned_model = model
    st.session_state.selected_checkpoint = checkpoint_name
    st.session_state.test_models_loaded = True



def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256, temperature: float = 0.7, template: str = None):
    """Generate a response from the model."""
    from mlx_lm import generate
    
    # Apply template if provided
    formatted_prompt = prompt
    if template and "{response}" in template:
        # Extract the part before the response placeholder
        # This gives us the prompt structure ending right where the model should start generating
        prompt_structure = template.split("{response}")[0]
        try:
            formatted_prompt = prompt_structure.format(instruction=prompt)
        except Exception as e:
            print(f"Error formatting prompt with template: {e}")
            formatted_prompt = prompt
            
    print(f"Generating with prompt: {formatted_prompt!r}")
    
    response = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return response


def render_chat_comparison_panel():
    """Render the dual chat comparison panel."""
    st.markdown('<h2 class="section-header">💬 Chat Comparison</h2>', unsafe_allow_html=True)
    st.markdown("Compare responses from the base model vs the fine-tuned model side by side.")
    
    config = st.session_state.config
    
    # Checkpoint selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        checkpoints = get_available_checkpoints()
        if not checkpoints:
            st.warning("⚠️ No checkpoints found. Train a model first!")
            return
        
        selected = st.selectbox(
            "📁 Select Checkpoint",
            options=checkpoints,
            index=checkpoints.index("best") if "best" in checkpoints else 0,
            help="Choose which checkpoint to load for the fine-tuned model"
        )
    
    with col2:
        if st.button("📥 Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models... This may take a while."):
                try:
                    load_test_models(selected)
                    st.success("✅ Models loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading models: {e}")
    
    with col3:
        if st.button("🗑️ Unload", use_container_width=True, disabled=not st.session_state.test_models_loaded):
            st.session_state.test_base_model = None
            st.session_state.test_finetuned_model = None
            st.session_state.test_tokenizer = None
            st.session_state.test_models_loaded = False
            st.session_state.test_chat_history = []
            st.rerun()
    
    # Model status
    if st.session_state.test_models_loaded:
        st.success(f"✅ Models loaded from checkpoint: **{st.session_state.selected_checkpoint}**")
    else:
        st.info("💡 Click 'Load Models' to start comparing responses.")
        return
    
    st.divider()
    
    # Generation parameters
    with st.expander("⚙️ Generation Parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            max_tokens = st.slider("Max Tokens", 32, 512, 256, step=32)
        with col2:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        with col3:
            st.markdown("**Model:**")
            st.code(config.model.name.split('/')[-1])
    
    # Input field
    user_input = st.text_area(
        "💭 Enter your question or prompt:",
        placeholder="Ask a question to compare how the base and fine-tuned models respond...",
        height=100,
        key="test_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True, disabled=not user_input)
    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.test_chat_history = []
            st.rerun()
    
    if generate_btn and user_input:
        with st.spinner("Generating responses..."):
            try:
                # Generate from base model
                base_response = generate_response(
                    st.session_state.test_base_model,
                    st.session_state.test_tokenizer,
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    template=config.data.prompt_template
                )
                
                # Generate from fine-tuned model
                finetuned_response = generate_response(
                    st.session_state.test_finetuned_model,
                    st.session_state.test_tokenizer,
                    user_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    template=config.data.prompt_template
                )
                
                # Add to history
                st.session_state.test_chat_history.append({
                    "prompt": user_input,
                    "base_response": base_response,
                    "finetuned_response": finetuned_response
                })
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
    
    # Display chat history
    if st.session_state.test_chat_history:
        for i, entry in enumerate(reversed(st.session_state.test_chat_history)):
            st.markdown(f"### 💭 Prompt {len(st.session_state.test_chat_history) - i}")
            st.markdown(f"**{entry['prompt']}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔵 Base Model")
                st.markdown(f"""
                <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 10px; padding: 16px;">
                {entry['base_response']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🟢 Fine-tuned Model")
                st.markdown(f"""
                <div style="background: rgba(78, 205, 196, 0.1); border: 1px solid rgba(78, 205, 196, 0.3); border-radius: 10px; padding: 16px;">
                {entry['finetuned_response']}
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()


def render_training_metrics_panel():
    """Render the training metrics visualization panel."""
    st.markdown('<h2 class="section-header">📈 Training Metrics</h2>', unsafe_allow_html=True)
    st.markdown("Visualize training progress and configuration parameters.")
    
    config = st.session_state.config
    checkpoints = get_available_checkpoints()
    
    if not checkpoints:
        st.warning("⚠️ No checkpoints found. Train a model first!")
        return
    
    # Try to load training logs
    log_file = PROJECT_ROOT / "outputs" / "logs" / "training_log.jsonl"
    training_logs = []
    train_start_info = None
    train_end_info = None
    step_entries = []
    eval_entries = []
    epoch_entries = []
    
    if log_file.exists():
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        training_logs.append(entry)
                        entry_type = entry.get("type", "")
                        if entry_type == "train_start":
                            train_start_info = entry
                        elif entry_type == "train_end":
                            train_end_info = entry
                        elif entry_type == "step":
                            step_entries.append(entry)
                        elif entry_type == "eval":
                            eval_entries.append(entry)
                        elif entry_type == "epoch_end":
                            epoch_entries.append(entry)
        except Exception as e:
            st.warning(f"Could not load training logs: {e}")
    
    has_logs = len(step_entries) > 0
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown("### ⚙️ Training Configuration")
        
        # Show training config from logs if available, otherwise from current config
        if train_start_info:
            log_config = train_start_info.get("training_config", {})
            log_lora = train_start_info.get("lora_config", {})
            model_name = train_start_info.get("model_name", config.model.name)
            
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | **Model** | `{model_name}` |
            | **LoRA Rank** | {log_lora.get('rank', config.lora.rank)} |
            | **LoRA Alpha** | {log_lora.get('alpha', config.lora.alpha)} |
            | **Dropout** | {log_lora.get('dropout', config.lora.dropout)} |
            | **Learning Rate** | {log_config.get('learning_rate', config.training.learning_rate):.2e} |
            | **Batch Size** | {log_config.get('batch_size', config.training.batch_size)} |
            | **Epochs** | {log_config.get('num_epochs', config.training.num_epochs)} |
            | **Warmup Steps** | {log_config.get('warmup_steps', config.training.warmup_steps)} |
            | **Max Seq Length** | {log_config.get('max_seq_length', config.model.max_seq_length)} |
            """)
            
            # Show training data info
            train_samples = train_start_info.get("train_samples", 0)
            val_samples = train_start_info.get("val_samples", 0)
            if train_samples or val_samples:
                st.markdown(f"**📊 Data:** {train_samples} train / {val_samples} validation samples")
        else:
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | **Model** | `{config.model.name}` |
            | **LoRA Rank** | {config.lora.rank} |
            | **LoRA Alpha** | {config.lora.alpha} |
            | **Dropout** | {config.lora.dropout} |
            | **Learning Rate** | {config.training.learning_rate:.2e} |
            | **Batch Size** | {config.training.batch_size} |
            | **Epochs** | {config.training.num_epochs} |
            | **Warmup Steps** | {config.training.warmup_steps} |
            | **Max Seq Length** | {config.model.max_seq_length} |
            """)
        
        st.markdown("### 💾 Available Checkpoints")
        for cp in checkpoints:
            adapter_path = PROJECT_ROOT / "outputs" / "checkpoints" / cp / "adapters.safetensors"
            if adapter_path.exists():
                size_mb = adapter_path.stat().st_size / (1024 * 1024)
                icon = "⭐" if cp == "best" else "🏁" if cp == "final" else "📌"
                st.markdown(f"{icon} **{cp}** — {size_mb:.1f} MB")
    
    with col2:
        st.markdown("### 📊 Training Progress")
        
        # Try to load training state from best/final checkpoint
        for cp_name in ["final", "best"]:
            state_path = PROJECT_ROOT / "outputs" / "checkpoints" / cp_name / "trainer_state.json"
            if state_path.exists():
                try:
                    with open(state_path, "r") as f:
                        state = json.load(f)
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Total Steps", state.get("global_step", "N/A"))
                    with col_b:
                        st.metric("Epochs Completed", state.get("epoch", "N/A") + 1 if isinstance(state.get("epoch"), int) else "N/A")
                    with col_c:
                        best_loss = state.get("best_val_loss")
                        if best_loss and best_loss != float("inf"):
                            st.metric("Best Val Loss", f"{best_loss:.4f}")
                        else:
                            st.metric("Best Val Loss", "N/A")
                    
                    st.success(f"✅ Loaded training state from **{cp_name}** checkpoint")
                    break
                except Exception as e:
                    st.warning(f"Could not load training state: {e}")
        else:
            st.info("💡 No training state file found. Run training to generate metrics.")
        
        # Loss Curve from training logs
        st.markdown("### 📉 Loss Curve")
        if has_logs:
            import pandas as pd
            
            # Create dataframe for step losses
            steps = [e["step"] for e in step_entries]
            losses = [e["loss"] for e in step_entries]
            
            df_loss = pd.DataFrame({
                "Step": steps,
                "Training Loss": losses,
            })
            
            # Add validation losses if available
            if eval_entries:
                eval_steps = [e["step"] for e in eval_entries]
                eval_losses = [e["val_loss"] for e in eval_entries]
                df_val = pd.DataFrame({
                    "Step": eval_steps,
                    "Validation Loss": eval_losses,
                })
                # Merge with training losses
                df_loss = df_loss.merge(df_val, on="Step", how="outer").sort_values("Step")
            
            # Display loss chart
            st.line_chart(df_loss.set_index("Step"))
            
            # Show epoch markers
            if epoch_entries:
                epoch_info = ", ".join([f"Epoch {e['epoch']+1}: step {e['global_step']}" for e in epoch_entries])
                st.caption(f"📌 Epoch markers: {epoch_info}")
        else:
            st.info("💡 Loss curve visualization requires training log data. Run training with the built-in trainer to generate detailed logs.")
        
        # Learning Rate Curve
        if has_logs:
            st.markdown("### 📈 Learning Rate")
            lr_steps = [e["step"] for e in step_entries]
            lr_values = [e["learning_rate"] for e in step_entries]
            
            import pandas as pd
            df_lr = pd.DataFrame({
                "Step": lr_steps,
                "Learning Rate": lr_values,
            })
            st.line_chart(df_lr.set_index("Step"))
        
        # Show checkpoint timeline
        st.markdown("### 🕐 Checkpoint Timeline")
        step_checkpoints = [cp for cp in checkpoints if cp.startswith("step-")]
        if step_checkpoints:
            import pandas as pd
            steps = []
            for cp in step_checkpoints:
                try:
                    step_num = int(cp.split("-")[1])
                    steps.append({"Checkpoint": cp, "Step": step_num})
                except:
                    pass
            if steps:
                df = pd.DataFrame(steps)
                st.bar_chart(df.set_index("Checkpoint")["Step"])
        else:
            st.info("No step checkpoints found yet.")
    
    # Additional training details section
    if has_logs:
        with st.expander("📊 Detailed Training Metrics", expanded=False):
            import pandas as pd
            
            # Create comprehensive metrics table
            df_metrics = pd.DataFrame(step_entries)
            cols_to_show = ["step", "epoch", "loss", "learning_rate", "tokens_per_second", "elapsed_time"]
            available_cols = [c for c in cols_to_show if c in df_metrics.columns]
            
            if available_cols:
                df_display = df_metrics[available_cols].copy()
                df_display.columns = ["Step", "Epoch", "Loss", "Learning Rate", "Tokens/sec", "Time (s)"][:len(available_cols)]
                
                # Format numeric columns
                if "Loss" in df_display.columns:
                    df_display["Loss"] = df_display["Loss"].apply(lambda x: f"{x:.4f}")
                if "Learning Rate" in df_display.columns:
                    df_display["Learning Rate"] = df_display["Learning Rate"].apply(lambda x: f"{x:.2e}")
                if "Time (s)" in df_display.columns:
                    df_display["Time (s)"] = df_display["Time (s)"].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Training summary
            if train_end_info:
                st.markdown("#### 🏁 Training Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Steps", train_end_info.get("total_steps", "N/A"))
                with col2:
                    total_time = train_end_info.get("total_time", 0)
                    if total_time:
                        mins = int(total_time // 60)
                        secs = int(total_time % 60)
                        st.metric("Training Time", f"{mins}m {secs}s")
                    else:
                        st.metric("Training Time", "N/A")
                with col3:
                    final_loss = train_end_info.get("final_loss")
                    st.metric("Final Loss", f"{final_loss:.4f}" if final_loss else "N/A")
                with col4:
                    best_val = train_end_info.get("best_val_loss")
                    if best_val and best_val != float("inf"):
                        st.metric("Best Val Loss", f"{best_val:.4f}")
                    else:
                        st.metric("Best Val Loss", "N/A")


def render_batch_testing_panel():
    """Render the batch testing panel."""
    st.markdown('<h2 class="section-header">📋 Batch Testing</h2>', unsafe_allow_html=True)
    st.markdown("Run automated testing on multiple questions at once.")
    
    if not st.session_state.test_models_loaded:
        st.warning("⚠️ Load models first using the Chat Comparison tab!")
        return
    
    # Input options
    input_method = st.radio(
        "📥 Input Method",
        options=["📝 Manual Input", "📄 Upload JSONL"],
        horizontal=True
    )
    
    test_questions = []
    
    if input_method == "📝 Manual Input":
        st.markdown("Enter one question per line:")
        manual_input = st.text_area(
            "Questions",
            placeholder="What is machine learning?\nExplain neural networks.\nHow does gradient descent work?",
            height=150,
            label_visibility="collapsed"
        )
        if manual_input:
            test_questions = [q.strip() for q in manual_input.split("\n") if q.strip()]
    
    else:
        uploaded_file = st.file_uploader(
            "Upload JSONL file",
            type=["jsonl", "json"],
            help="Each line should be a JSON object with an 'instruction' or 'question' field"
        )
        if uploaded_file:
            try:
                content = uploaded_file.read().decode("utf-8")
                for line in content.split("\n"):
                    if line.strip():
                        item = json.loads(line)
                        question = item.get("instruction") or item.get("question") or item.get("prompt") or item.get("text")
                        if question:
                            test_questions.append(question)
                st.success(f"✅ Loaded {len(test_questions)} questions")
            except Exception as e:
                st.error(f"❌ Error parsing file: {e}")
    
    if test_questions:
        st.info(f"📝 **{len(test_questions)}** questions ready for testing")
    
    # Generation parameters
    with st.expander("⚙️ Generation Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            batch_max_tokens = st.slider("Max Tokens", 32, 512, 128, step=32, key="batch_max_tokens")
        with col2:
            batch_temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1, key="batch_temp")
    
    # Run batch
    col1, col2 = st.columns(2)
    with col1:
        run_batch = st.button(
            "🚀 Run Batch Testing",
            type="primary",
            use_container_width=True,
            disabled=len(test_questions) == 0
        )
    
    if run_batch and test_questions:
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, question in enumerate(test_questions):
            status_text.markdown(f"Processing question {i+1}/{len(test_questions)}...")
            progress_bar.progress((i + 1) / len(test_questions))
            
            try:
                # Generate from fine-tuned model only for batch testing
                response = generate_response(
                    st.session_state.test_finetuned_model,
                    st.session_state.test_tokenizer,
                    question,
                    max_tokens=batch_max_tokens,
                    temperature=batch_temperature
                )
                results.append({
                    "Question": question,
                    "Response": response,
                    "Status": "✅"
                })
            except Exception as e:
                results.append({
                    "Question": question,
                    "Response": f"Error: {e}",
                    "Status": "❌"
                })
        
        status_text.markdown("✅ **Batch testing complete!**")
        
        # Display results
        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Export options
        st.markdown("### 📤 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name="batch_test_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            jsonl_data = "\n".join([json.dumps(r) for r in results])
            st.download_button(
                "📥 Download JSONL",
                data=jsonl_data,
                file_name="batch_test_results.jsonl",
                mime="application/json",
                use_container_width=True
            )


def page_testing():
    """Render model testing page with three panels."""
    st.markdown('<h1 class="main-title">🧪 Model Testing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Compare base vs fine-tuned model • Batch testing • Visualize training metrics</p>', unsafe_allow_html=True)
    
    # Three tabs for the panels - Training Metrics is now the third tab
    tab1, tab2, tab3 = st.tabs(["💬 Chat Comparison", "📋 Batch Testing", "📈 Training Metrics"])
    
    with tab1:
        render_chat_comparison_panel()
    
    with tab2:
        render_batch_testing_panel()
    
    with tab3:
        render_training_metrics_panel()


# ============================================================================
# Page: HuggingFace Upload
# ============================================================================

def page_upload():
    """Render HuggingFace upload page with comprehensive model metadata support."""
    st.markdown('<h1 class="main-title">☁️ HuggingFace Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Publish your fine-tuned models and checkpoints to the HuggingFace Hub</p>', unsafe_allow_html=True)
    
    config = st.session_state.config
    
    # =========================================================================
    # Authentication Section
    # =========================================================================
    st.markdown('<h2 class="section-header">🔐 Authentication</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        hf_token = st.text_input(
            "🔑 HuggingFace Token",
            value=get_hf_token() or "",
            type="password",
            help="Token with write permissions from huggingface.co/settings/tokens"
        )
        
        if hf_token:
            st.markdown('<span class="badge badge-success">✓ Token configured</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warning">⚠ Token required</span>', unsafe_allow_html=True)
    
    with col2:
        repo_id = st.text_input(
            "📦 Repository ID",
            value=config.huggingface.repo_id or "",
            placeholder="username/model-name",
            help="Format: username/model-name"
        )
        
        private = st.checkbox("🔒 Private Repository", value=config.huggingface.private)
    
    # Check repository status
    if repo_id and hf_token:
        repo_info = check_repo_exists(repo_id, hf_token)
        if repo_info.get("exists"):
            st.markdown(f'''
            <div style="background: rgba(255, 142, 83, 0.15); border: 1px solid rgba(255, 142, 83, 0.3); border-radius: 10px; padding: 12px; margin: 8px 0;">
                <span style="color: #FF8E53; font-weight: 600;">🔄 Update Existing Repository</span><br/>
                <span style="color: #9ca3af; font-size: 0.85rem;">
                    📅 Last modified: {repo_info.get("last_modified", "Unknown")[:10] if repo_info.get("last_modified") else "Unknown"} | 
                    📁 {repo_info.get("siblings", 0)} files | 
                    ⬇️ {repo_info.get("downloads", 0)} downloads
                </span>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-success">🆕 New Repository (will be created)</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # =========================================================================
    # Model Card Section (NEW)
    # =========================================================================
    st.markdown('<h2 class="section-header">📋 Model Card Information</h2>', unsafe_allow_html=True)
    st.caption("Fill in model metadata to generate a comprehensive README for your HuggingFace repository")
    
    # Initialize session state for model card fields
    if 'model_card' not in st.session_state:
        st.session_state.model_card = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_description = st.text_area(
            "📝 Model Description",
            value=st.session_state.model_card.get('description', ''),
            placeholder="Describe what this model does, its purpose, and capabilities...",
            height=100,
            help="A brief description of your fine-tuned model"
        )
        st.session_state.model_card['description'] = model_description
        
        license_options = [
            "MIT", "Apache-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "CC-BY-NC-4.0",
            "GPL-3.0", "BSD-3-Clause", "OpenRAIL", "Other"
        ]
        license_selected = st.selectbox(
            "📄 License",
            options=license_options,
            index=license_options.index(st.session_state.model_card.get('license', 'MIT')) if st.session_state.model_card.get('license') in license_options else 0,
            help="License for your model"
        )
        st.session_state.model_card['license'] = license_selected
        
        base_model = st.text_input(
            "🏛️ Base Model",
            value=st.session_state.model_card.get('base_model', config.model.name),
            placeholder="e.g., meta-llama/Llama-3.2-1B",
            help="The original model this was fine-tuned from"
        )
        st.session_state.model_card['base_model'] = base_model
    
    with col2:
        tags_input = st.text_input(
            "🏷️ Tags",
            value=st.session_state.model_card.get('tags', ''),
            placeholder="lora, fine-tuned, mlx, text-generation",
            help="Comma-separated tags for discoverability"
        )
        st.session_state.model_card['tags'] = tags_input
        
        language_options = ["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko", "ar", "ru", "nl", "multilingual", "other"]
        language_selected = st.selectbox(
            "🌐 Language",
            options=language_options,
            index=language_options.index(st.session_state.model_card.get('language', 'en')) if st.session_state.model_card.get('language') in language_options else 0,
            help="Primary language of the model/training data"
        )
        st.session_state.model_card['language'] = language_selected
        
        task_types = [
            "text-generation", "text2text-generation", "question-answering",
            "summarization", "translation", "conversational", "text-classification", "other"
        ]
        task_selected = st.selectbox(
            "🎯 Task Type",
            options=task_types,
            index=task_types.index(st.session_state.model_card.get('task', 'text-generation')) if st.session_state.model_card.get('task') in task_types else 0,
            help="Primary task this model is designed for"
        )
        st.session_state.model_card['task'] = task_selected
    
    # Expandable advanced fields
    with st.expander("📖 Additional Information (Optional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            training_data = st.text_area(
                "📚 Training Data Description",
                value=st.session_state.model_card.get('training_data', ''),
                placeholder="Describe the dataset used for fine-tuning...",
                height=80,
                help="Information about your training dataset"
            )
            st.session_state.model_card['training_data'] = training_data
            
            intended_uses = st.text_area(
                "✅ Intended Uses",
                value=st.session_state.model_card.get('intended_uses', ''),
                placeholder="What is this model intended to be used for?",
                height=80,
                help="Describe the intended use cases"
            )
            st.session_state.model_card['intended_uses'] = intended_uses
        
        with col2:
            limitations = st.text_area(
                "⚠️ Limitations & Biases",
                value=st.session_state.model_card.get('limitations', ''),
                placeholder="Known limitations, biases, or risks...",
                height=80,
                help="Important limitations users should know about"
            )
            st.session_state.model_card['limitations'] = limitations
            
            author = st.text_input(
                "👤 Author / Organization",
                value=st.session_state.model_card.get('author', ''),
                placeholder="Your name or organization",
                help="Who created this model"
            )
            st.session_state.model_card['author'] = author
    
    st.divider()
    
    # =========================================================================
    # Training Configuration Summary (NEW)
    # =========================================================================
    st.markdown('<h2 class="section-header">⚙️ Training Configuration</h2>', unsafe_allow_html=True)
    st.caption("This information will be included in your model card automatically")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("LoRA Rank", config.lora.rank)
    with col2:
        st.metric("LoRA Alpha", config.lora.alpha)
    with col3:
        st.metric("Batch Size", config.training.batch_size)
    with col4:
        st.metric("Learning Rate", f"{config.training.learning_rate:.1e}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Epochs", config.training.num_epochs)
    with col2:
        st.metric("Warmup", config.training.warmup_steps)
    with col3:
        st.metric("Gradient Accum", config.training.gradient_accumulation_steps)
    with col4:
        # Display target modules if available
        if hasattr(config.lora, 'target_modules') and config.lora.target_modules:
            target_count = len(config.lora.target_modules)
            st.metric("Target Modules", f"{target_count} layers")
        else:
            st.metric("Target Modules", "Auto")
    
    st.divider()
    
    # =========================================================================
    # Configuration Files Section (NEW)
    # =========================================================================
    st.markdown('<h2 class="section-header">📦 Include Configuration Files</h2>', unsafe_allow_html=True)
    st.caption("Optionally include configuration files with your model for reproducibility")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_training_config = st.checkbox(
            "⚙️ training_config.json",
            value=True,
            help="Training hyperparameters (learning rate, batch size, epochs, etc.)"
        )
        include_lora_config = st.checkbox(
            "🔧 lora_config.json",
            value=True,
            help="LoRA adapter settings (rank, alpha, target modules)"
        )
    
    with col2:
        include_data_config = st.checkbox(
            "📚 data_config.json",
            value=True,
            help="Data paths and prompt template used for training"
        )
        include_full_config = st.checkbox(
            "📋 full_config.yaml",
            value=False,
            help="Complete configuration file (all settings)"
        )
    
    st.divider()
    
    # =========================================================================
    # Upload Options
    # =========================================================================
    st.markdown('<h2 class="section-header">📤 Upload Options</h2>', unsafe_allow_html=True)
    
    upload_type = st.radio(
        "What would you like to upload?",
        options=["Final Model", "Specific Checkpoint"],
        horizontal=True
    )
    
    if upload_type == "Final Model":
        model_path = st.text_input(
            "📁 Model Path",
            value=str(Path(config.output.checkpoints_dir) / "final"),
            help="Directory containing the final adapter weights"
        )
        
        path_to_upload = Path(model_path)
        
    else:  # Checkpoint
        checkpoints_dir = Path(config.output.checkpoints_dir)
        
        if checkpoints_dir.exists():
            checkpoints = [d.name for d in checkpoints_dir.iterdir() if d.is_dir()]
            if checkpoints:
                selected_checkpoint = st.selectbox(
                    "📁 Select Checkpoint",
                    options=checkpoints
                )
                path_to_upload = checkpoints_dir / selected_checkpoint
            else:
                st.warning("No checkpoints available")
                path_to_upload = None
        else:
            st.warning("Checkpoints directory does not exist")
            path_to_upload = None
    
    # Enhanced file preview with size aggregation
    if path_to_upload and path_to_upload.exists():
        with st.expander("📄 Files to Upload", expanded=True):
            files = list(path_to_upload.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            # Show total size
            if total_size > 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024**3):.2f} GB"
            elif total_size > 1024 * 1024:
                size_str = f"{total_size / (1024**2):.2f} MB"
            else:
                size_str = f"{total_size / 1024:.2f} KB"
            
            st.info(f"📦 **Total Size:** {size_str} | **Files:** {len(files)}")
            
            # Categorize files
            weight_files = [f for f in files if f.suffix in ['.safetensors', '.bin', '.pt', '.npz']]
            config_files = [f for f in files if f.suffix in ['.json', '.yaml', '.yml']]
            other_files = [f for f in files if f not in weight_files and f not in config_files]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🧠 Weights**")
                for f in weight_files[:5]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    st.text(f"  {f.name} ({size_mb:.1f} MB)")
                if len(weight_files) > 5:
                    st.text(f"  ... +{len(weight_files) - 5} more")
                elif not weight_files:
                    st.text("  None")
            
            with col2:
                st.markdown("**⚙️ Configs**")
                for f in config_files[:5]:
                    size_kb = f.stat().st_size / 1024
                    st.text(f"  {f.name} ({size_kb:.1f} KB)")
                if len(config_files) > 5:
                    st.text(f"  ... +{len(config_files) - 5} more")
                elif not config_files:
                    st.text("  None")
            
            with col3:
                st.markdown("**📄 Other**")
                for f in other_files[:5]:
                    size_kb = f.stat().st_size / 1024
                    st.text(f"  {f.name} ({size_kb:.1f} KB)")
                if len(other_files) > 5:
                    st.text(f"  ... +{len(other_files) - 5} more")
                elif not other_files:
                    st.text("  None")
    
    st.divider()
    
    # =========================================================================
    # README Generation (NEW)
    # =========================================================================
    st.markdown('<h2 class="section-header">📝 README Generation</h2>', unsafe_allow_html=True)
    
    # Initialize session state for AI-generated README
    if 'ai_generated_readme' not in st.session_state:
        st.session_state.ai_generated_readme = None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        readme_mode = st.radio(
            "Generation Method",
            options=["📄 Template (instant)", "🤖 AI-powered (OpenRouter)"],
            horizontal=True,
            help="Choose how to generate the README"
        )
    
    with col2:
        if "AI-powered" in readme_mode:
            if is_openrouter_configured():
                st.markdown('<span class="badge badge-success">✅ OpenRouter configured</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge badge-warning">⚠️ Add OPENROUTER_API_KEY to .env</span>', unsafe_allow_html=True)
    
    # Model selector for AI mode
    selected_ai_model = None
    if "AI-powered" in readme_mode:
        openrouter_config = get_openrouter_config()
        default_model = openrouter_config.get("model", "qwen/qwen3-0.6b-04-28")
        
        selected_ai_model = st.text_input(
            "🧠 OpenRouter Model",
            value=default_model,
            placeholder="e.g. xiaomi/mimo-v2-flash:free",
            help="Enter any OpenRouter model ID (e.g. anthropic/claude-3.5-sonnet, openai/gpt-4o-mini)"
        )
    
    # Build template README (used for both modes as fallback/base)
    tags_list = [t.strip() for t in tags_input.split(',') if t.strip()] if tags_input else ['lora', 'fine-tuned', 'mlx']
    tags_yaml = '\n'.join([f'- {tag}' for tag in tags_list])
    
    template_readme = f'''---
license: {license_selected.lower()}
language:
- {language_selected}
base_model: {base_model}
tags:
{tags_yaml}
pipeline_tag: {task_selected}
---

# {repo_id.split('/')[-1] if repo_id else 'Fine-tuned Model'}

{model_description if model_description else 'A fine-tuned language model using LoRA adapters on MLX.'}

## Model Details

- **Base Model:** [{base_model}](https://huggingface.co/{base_model})
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Framework:** MLX (Apple Silicon optimized)
- **License:** {license_selected}

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank | {config.lora.rank} |
| LoRA Alpha | {config.lora.alpha} |
| Batch Size | {config.training.batch_size} |
| Learning Rate | {config.training.learning_rate} |
| Epochs | {config.training.num_epochs} |
| Warmup Steps | {config.training.warmup_steps} |
| Gradient Accumulation | {config.training.gradient_accumulation_steps} |

{f"## Training Data{chr(10)}{chr(10)}{training_data}" if training_data else ""}

{f"## Intended Uses{chr(10)}{chr(10)}{intended_uses}" if intended_uses else ""}

{f"## Limitations{chr(10)}{chr(10)}{limitations}" if limitations else ""}

## Usage

```python
from mlx_lm import load, generate

# Load with LoRA adapters
model, tokenizer = load("{repo_id if repo_id else 'username/model-name'}", adapter_path="./adapters")

# Generate text
prompt = "Your prompt here"
response = generate(model, tokenizer, prompt=prompt, max_tokens=256)
print(response)
```

{f"---{chr(10)}{chr(10)}Created by {author}" if author else ""}
'''
    
    # AI-powered generation
    if "AI-powered" in readme_mode:
        if st.button("🤖 Generate README with AI", use_container_width=True, disabled=not is_openrouter_configured()):
            with st.spinner("Generating README with AI..."):
                try:
                    ai_prompt = f"""Generate a professional HuggingFace model card README in markdown format for this fine-tuned model.

MODEL INFORMATION:
- Base Model: {base_model}
- Repository: {repo_id if repo_id else 'username/model-name'}
- Task Type: {task_selected}
- Language: {language_selected}
- License: {license_selected}
- Description: {model_description if model_description else 'A fine-tuned language model'}
- Tags: {', '.join(tags_list)}

TRAINING CONFIGURATION:
- LoRA Rank: {config.lora.rank}
- LoRA Alpha: {config.lora.alpha}
- Batch Size: {config.training.batch_size}
- Learning Rate: {config.training.learning_rate}
- Epochs: {config.training.num_epochs}
- Framework: MLX (Apple Silicon optimized)

{f"Training Data: {training_data}" if training_data else ""}
{f"Intended Uses: {intended_uses}" if intended_uses else ""}
{f"Limitations: {limitations}" if limitations else ""}
{f"Author: {author}" if author else ""}

Generate a complete, professional README with:
1. YAML frontmatter (license, language, base_model, tags, pipeline_tag)
2. Title and description
3. Model details section
4. Training configuration table
5. Usage example with Python code for mlx_lm
6. Any relevant sections based on the provided info

Make it engaging and informative for users discovering this model."""

                    ai_readme = generate_with_openrouter(
                        prompt=ai_prompt,
                        model=selected_ai_model,
                        max_tokens=1500,
                        temperature=0.7
                    )
                    st.session_state.ai_generated_readme = ai_readme
                    st.success("✅ README generated with AI!")
                except Exception as e:
                    st.error(f"❌ AI generation failed: {e}")
                    st.session_state.ai_generated_readme = None
        
        # Show editable AI-generated README or placeholder
        if st.session_state.ai_generated_readme:
            readme_content = st.text_area(
                "📖 AI-Generated README (editable)",
                value=st.session_state.ai_generated_readme,
                height=400,
                help="Edit the AI-generated README before uploading"
            )
        else:
            st.info("👆 Click the button above to generate README with AI")
            readme_content = template_readme
    else:
        # Template mode - show preview
        readme_content = template_readme
        with st.expander("📖 README Preview", expanded=False):
            st.code(readme_content, language="markdown")
    
    st.divider()
    
    # =========================================================================
    # Actions
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        upload_disabled = not all([hf_token, repo_id, path_to_upload and path_to_upload.exists()])
        
        if st.button(
            "🚀 Upload to HuggingFace",
            type="primary",
            use_container_width=True,
            disabled=upload_disabled
        ):
            with st.spinner("Uploading model..."):
                try:
                    # Save README to the upload path
                    if path_to_upload and path_to_upload.exists() and readme_content:
                        readme_path = path_to_upload / "README.md"
                        with open(readme_path, 'w') as f:
                            f.write(readme_content)
                        st.info("📝 README.md added to upload")
                    
                    # Write configuration files if selected
                    import json
                    configs_added = []
                    
                    if include_training_config and path_to_upload and path_to_upload.exists():
                        training_config_path = path_to_upload / "training_config.json"
                        with open(training_config_path, 'w') as f:
                            json.dump(config.training.to_dict(), f, indent=2)
                        configs_added.append("training_config.json")
                    
                    if include_lora_config and path_to_upload and path_to_upload.exists():
                        lora_config_path = path_to_upload / "lora_config.json"
                        with open(lora_config_path, 'w') as f:
                            json.dump(config.lora.to_dict(), f, indent=2)
                        configs_added.append("lora_config.json")
                    
                    if include_data_config and path_to_upload and path_to_upload.exists():
                        data_config_path = path_to_upload / "data_config.json"
                        with open(data_config_path, 'w') as f:
                            json.dump(config.data.to_dict(), f, indent=2)
                        configs_added.append("data_config.json")
                    
                    if include_full_config and path_to_upload and path_to_upload.exists():
                        full_config_path = path_to_upload / "full_config.yaml"
                        config.to_yaml(str(full_config_path))
                        configs_added.append("full_config.yaml")
                    
                    if configs_added:
                        st.info(f"⚙️ Added config files: {', '.join(configs_added)}")
                    
                    if upload_type == "Final Model":
                        url = upload_model(
                            model_path=str(path_to_upload),
                            repo_id=repo_id,
                            token=hf_token,
                            private=private,
                        )
                    else:
                        url = upload_checkpoint(
                            checkpoint_path=str(path_to_upload),
                            repo_id=repo_id,
                            token=hf_token,
                            private=private,
                        )
                    
                    st.success(f"""
                    ✅ **Upload Complete!**
                    
                    🔗 [View on HuggingFace](https://huggingface.co/{repo_id})
                    """)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        if st.button("📋 List Remote Checkpoints", use_container_width=True, disabled=not repo_id):
            if repo_id and hf_token:
                with st.spinner("Fetching repository info..."):
                    try:
                        checkpoints = list_checkpoints(repo_id, token=hf_token)
                        if checkpoints:
                            st.info("**Checkpoints in repository:**\n" + "\n".join(f"- {cp}" for cp in checkpoints))
                        else:
                            st.info("No checkpoints found in repository")
                    except Exception as e:
                        st.error(f"Error: {e}")


# ============================================================================
# Main Navigation
# ============================================================================

def main():
    """Main application entry point."""
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🚀 MLX LoRA")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            options=["🏠 Home", "📊 Prepare Data", "🚀 Train", "🧪 Test Model", "☁️ HuggingFace"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick info panel
        st.markdown("### ℹ️ Quick Info")
        model_short = st.session_state.config.model.name.split('/')[-1]
        st.markdown(f"""
        **Model:** `{model_short}`
        
        **LoRA:** r={st.session_state.config.lora.rank}, α={st.session_state.config.lora.alpha}
        
        **Batch:** {st.session_state.config.training.batch_size}
        """)
        
        # Training status indicator
        if st.session_state.training_running:
            st.markdown("---")
            st.markdown("### 🔄 Status")
            st.warning("⏳ Training in progress...")
        
        # Theme toggle at bottom
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🌙" if st.session_state.theme == 'light' else "☀️", 
                        help="Switch to Light Mode" if st.session_state.theme == 'dark' else "Switch to Dark Mode",
                        key="theme_toggle"):
                st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
                st.rerun()
    
    # Route to appropriate page
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Prepare Data":
        page_data_preparation()
    elif page == "🚀 Train":
        page_training()
    elif page == "🧪 Test Model":
        page_testing()
    elif page == "☁️ HuggingFace":
        page_upload()


if __name__ == "__main__":
    main()
