"""
File Utilities
==============
Functions for finding and managing model files
"""

from pathlib import Path
import sys

# Get parent directory
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


def find_model_weights():
    """
    Find available model weights in runs/train directory
    Only returns cat_original and cat_enhanced models
    
    Returns:
        Dictionary of {model_name: model_path}
    """
    models = {}
    
    # Look in parent directory for trained models (only original and enhanced)
    runs_dir = parent_dir / "runs" / "train"
    
    if runs_dir.exists():
        for model_dir in runs_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                # Only include cat_original and cat_enhanced
                if model_name in ["cat_original", "cat_enhanced"]:
                    weights_path = model_dir / "weights" / "best.pt"
                    if weights_path.exists():
                        # Format display name
                        display_name = model_name.replace("_", " ").title()
                        models[display_name] = str(weights_path)
    
    return models


def get_parent_dir():
    """Get the parent directory path"""
    return parent_dir


def get_model_info(model_path):
    """
    Get model information from the training results
    
    Args:
        model_path: Path to the model weights
        
    Returns:
        Dictionary with model information
    """
    import os
    
    model_dir = Path(model_path).parent.parent
    info = {}
    
    # Try to read results.csv for metrics
    results_csv = model_dir / "results.csv"
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            if len(df) > 0:
                last_row = df.iloc[-1]
                info['epochs_trained'] = len(df)
                info['final_map50'] = last_row.get('metrics/mAP50(B)', 0)
                info['final_map50_95'] = last_row.get('metrics/mAP50-95(B)', 0)
                info['precision'] = last_row.get('metrics/precision(B)', 0)
                info['recall'] = last_row.get('metrics/recall(B)', 0)
        except:
            pass
    
    # Get model file size
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        info['size_mb'] = f"{size_mb:.1f}"
    
    return info
