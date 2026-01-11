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


def load_test_metrics(model_path):
    """
    Load test set metrics from test_set_metrics.txt if available
    
    Args:
        model_path: Path to the model weights (best.pt)
        
    Returns:
        Dictionary with test metrics or None if not available
    """
    metrics_file = Path(model_path).parent / "test_set_metrics.txt"
    
    if not metrics_file.exists():
        return None
    
    metrics = {}
    try:
        with open(metrics_file, 'r') as f:
            content = f.read()
            
        # Parse metrics from file
        for line in content.split('\n'):
            if 'mAP@0.5      :' in line:
                metrics['map50'] = float(line.split(':')[1].strip())
            elif 'mAP@0.5:0.95 :' in line:
                metrics['map50_95'] = float(line.split(':')[1].strip())
            elif 'Precision    :' in line:
                metrics['precision'] = float(line.split(':')[1].strip())
            elif 'Recall       :' in line:
                metrics['recall'] = float(line.split(':')[1].strip())
                
        return metrics if metrics else None
    except Exception as e:
        return None
