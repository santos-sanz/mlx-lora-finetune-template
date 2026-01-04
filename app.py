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
from src.data_utils import load_dataset, convert_to_mlx_format
from src.hf_utils import get_hf_token, upload_model, upload_checkpoint, list_checkpoints


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

# Premium dark theme with high contrast
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background - solid dark for contrast */
    .stApp {
        background: linear-gradient(160deg, #0a0a12 0%, #12121f 40%, #1a1a2e 70%, #12121f 100%);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d15 0%, #151520 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.6rem;
        letter-spacing: -0.02em;
    }
    
    /* HIGH CONTRAST TEXT - All text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    .stMarkdown td, .stMarkdown th {
        color: #f1f5f9 !important;
    }
    
    .stMarkdown strong, .stMarkdown b {
        color: #ffffff !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
    }
    
    /* Captions and help text */
    .stCaption, [data-testid="stCaption"] {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }
    
    /* Help tooltips */
    [data-testid="stTooltipIcon"] {
        color: #a5b4fc !important;
    }
    
    /* Labels with high contrast */
    label, .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stSlider label, .stCheckbox label, .stRadio label, .stMultiSelect label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Input placeholder text */
    input::placeholder, textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Expander header text */
    .streamlit-expanderHeader p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    /* Tab panel glass effect */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(20, 20, 35, 0.8);
        border-radius: 20px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 28px;
        backdrop-filter: blur(20px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 15, 25, 0.9);
        border-radius: 14px;
        padding: 8px;
        gap: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: #94a3b8 !important;
        font-weight: 600;
        padding: 14px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.2);
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.4);
    }
    
    /* Button styling - premium gradient */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white !important;
        border: none;
        border-radius: 14px;
        padding: 14px 32px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 40px rgba(99, 102, 241, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Primary button - emerald gradient */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 16px 40px rgba(16, 185, 129, 0.5);
    }
    
    /* Input fields - better contrast */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > textarea {
        background: rgba(15, 15, 25, 0.9) !important;
        border: 2px solid rgba(100, 116, 139, 0.4) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 1rem !important;
        padding: 14px 16px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > textarea:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.2) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #cbd5e1 !important;
    }
    
    /* Select box */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(15, 15, 25, 0.9) !important;
        border: 2px solid rgba(100, 116, 139, 0.4) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    /* Metric cards - glass morphism with glow */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 30, 50, 0.9) 0%, rgba(20, 20, 35, 0.9) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 24px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetric"]:hover {
        border-color: rgba(129, 140, 248, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Main title - vibrant gradient with glow */
    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #c084fc 60%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        text-shadow: 0 0 60px rgba(129, 140, 248, 0.5);
    }
    
    .subtitle {
        color: #cbd5e1 !important;
        font-size: 1.15rem;
        font-weight: 500;
        margin-bottom: 2.5rem;
    }
    
    /* Section headers - high contrast */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        margin: 2rem 0 1.2rem 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    /* Status badges - brighter colors */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 24px;
        font-size: 0.9rem;
        font-weight: 700;
    }
    
    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399 !important;
        border: 2px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-warning {
        background: rgba(251, 191, 36, 0.2);
        color: #fcd34d !important;
        border: 2px solid rgba(251, 191, 36, 0.4);
    }
    
    .badge-error {
        background: rgba(244, 63, 94, 0.2);
        color: #fb7185 !important;
        border: 2px solid rgba(244, 63, 94, 0.4);
    }
    
    .badge-info {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc !important;
        border: 2px solid rgba(99, 102, 241, 0.4);
    }
    
    /* Feature cards - premium glass with border glow */
    .feature-card {
        background: linear-gradient(145deg, rgba(30, 30, 50, 0.95) 0%, rgba(20, 20, 35, 0.95) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 24px;
        padding: 32px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
    }
    
    .feature-card:hover {
        border-color: rgba(139, 92, 246, 0.6);
        transform: translateY(-8px);
        box-shadow: 0 24px 48px rgba(99, 102, 241, 0.25);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 20px;
        filter: drop-shadow(0 4px 12px rgba(99, 102, 241, 0.3));
    }
    
    .feature-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-bottom: 12px;
    }
    
    .feature-desc {
        color: #cbd5e1 !important;
        font-size: 1rem;
        line-height: 1.7;
    }
    
    /* Info boxes - higher contrast */
    .stInfo {
        background: rgba(99, 102, 241, 0.15) !important;
        border: 2px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 14px !important;
        color: #e0e7ff !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        border: 2px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 14px !important;
        color: #d1fae5 !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.15) !important;
        border: 2px solid rgba(251, 191, 36, 0.4) !important;
        border-radius: 14px !important;
        color: #fef3c7 !important;
    }
    
    .stError {
        background: rgba(244, 63, 94, 0.15) !important;
        border: 2px solid rgba(244, 63, 94, 0.4) !important;
        border-radius: 14px !important;
        color: #fecdd3 !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: rgba(10, 10, 18, 0.95) !important;
        border: 1px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 14px !important;
    }
    
    .stCodeBlock code {
        color: #e2e8f0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(20, 20, 35, 0.9) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(100, 116, 139, 0.3) !important;
        margin: 2.5rem 0 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
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
        background: rgba(10, 10, 18, 0.95) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Load default config if not loaded
    if st.session_state.config is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        if config_path.exists():
            st.session_state.config = Config.from_yaml(str(config_path))
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
    )
    
    # Two main tabs
    tab1, tab2 = st.tabs(["📋 Structured JSON", "📝 Raw Text / Folder"])
    
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
        with st.expander("🤖 AI-Enhanced Generation (Optional)"):
            st.markdown("Use a small LLM to generate intelligent Q&A pairs or summaries.")
            
            use_ai = st.checkbox("✨ Enable AI Enhancement", value=False, help="Use Qwen3-0.6B for smarter data generation")
            
            if use_ai:
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
                    ai_format = st.radio("AI Generation", ["qa", "summary"], format_func=lambda x: {"qa": "❓ Q&A", "summary": "📝 Summary"}[x], horizontal=True)
                else:
                    st.warning("⚠️ Load AI model first to enable")
                    ai_format = "qa"
        
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
                    
                    # Process folder if applicable
                    if files_to_process:
                        for i, file_path in enumerate(files_to_process):
                            progress_bar.progress((i + 1) / len(files_to_process))
                            status_text.text(f"Processing: {file_path.name}")
                            
                            with open(file_path, "r", encoding="utf-8") as f:
                                text = f.read()
                            
                            if use_ai and st.session_state.helper_model:
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
                        if use_ai and st.session_state.helper_model:
                            def progress_cb(current, total):
                                progress_bar.progress((current + 1) / total)
                                status_text.text(f"Processing chunk {current + 1}/{total}...")
                            
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
        | Model | Size | Best For |
        |-------|------|----------|
        | `meta-llama/Llama-3.2-1B` | 1B | Fast testing, quick iterations |
        | `meta-llama/Llama-3.2-3B` | 3B | Balanced quality and speed |
        | `Qwen/Qwen2.5-1.5B` | 1.5B | Excellent multilingual support |
        | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Very efficient, good for chat |
        | `google/gemma-2-2b` | 2B | Google's compact model |
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
        
        if process.poll() is None:
            try:
                for line in iter(process.stdout.readline, ''):
                    if line:
                        st.session_state.training_logs.append(line.strip())
                        if len(st.session_state.training_logs) > 100:
                            st.session_state.training_logs = st.session_state.training_logs[-100:]
                    else:
                        break
            except:
                pass
            
            time.sleep(0.5)
            st.rerun()
        else:
            remaining = process.stdout.read()
            if remaining:
                st.session_state.training_logs.extend(remaining.strip().split('\n'))
            st.session_state.training_running = False
            st.success("✅ Training completed!")
    
    if st.session_state.training_logs:
        log_container.code('\n'.join(st.session_state.training_logs[-50:]), language="text")
    else:
        log_container.info("💡 Training logs will appear here when you start training...")


# ============================================================================
# Page: HuggingFace Upload
# ============================================================================

def page_upload():
    """Render HuggingFace upload page."""
    st.markdown('<h1 class="main-title">☁️ HuggingFace Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Publish your fine-tuned models and checkpoints to the HuggingFace Hub</p>', unsafe_allow_html=True)
    
    config = st.session_state.config
    
    # Authentication
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
    
    st.divider()
    
    # Upload options
    st.markdown('<h2 class="section-header">📤 Upload Options</h2>', unsafe_allow_html=True)
    
    upload_type = st.radio(
        "What would you like to upload?",
        options=["Final Model", "Specific Checkpoint"],
        horizontal=True
    )
    
    if upload_type == "Final Model":
        model_path = st.text_input(
            "📁 Model Path",
            value=str(Path(config.output.adapters_dir) / "final"),
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
    
    # Preview files
    if path_to_upload and path_to_upload.exists():
        with st.expander("📄 Files to Upload"):
            files = list(path_to_upload.glob("*"))
            for f in files[:10]:
                size_mb = f.stat().st_size / (1024 * 1024)
                st.text(f"  {f.name} ({size_mb:.2f} MB)")
            if len(files) > 10:
                st.text(f"  ... and {len(files) - 10} more files")
    
    st.divider()
    
    # Actions
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
            options=["🏠 Home", "📊 Prepare Data", "🚀 Train", "☁️ HuggingFace"],
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
    
    # Route to appropriate page
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Prepare Data":
        page_data_preparation()
    elif page == "🚀 Train":
        page_training()
    elif page == "☁️ HuggingFace":
        page_upload()


if __name__ == "__main__":
    main()
