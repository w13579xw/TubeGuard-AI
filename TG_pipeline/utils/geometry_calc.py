import cv2
import numpy as np

def calculate_geometry_metrics(mask):
    """
    Calculates geometric metrics from a binary mask of the pipe.
    
    Args:
        mask (np.ndarray): Binary mask (0: background, 255: pipe).
        
    Returns:
        dict: {
            "diameter": float,
            "wall_thickness": float,
            "eccentricity": float,
            "valid": bool
        }
    """
    metrics = {
        "diameter": 0.0,
        "wall_thickness": 0.0,
        "eccentricity": 0.0,
        "valid": False
    }
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return metrics
        
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Assuming the largest contour is the Outer Wall
    outer_cnt = contours[0]
    
    # Find the second largest contour which should be the Inner Wall (hole)
    # Note: cv2.findContours with RETR_TREE gives hierarchy, but sorting by area is a simple heuristic
    # for a pipe where the hole is the second largest object (or the hole is a child).
    # If the mask is a solid filled circle, we can't calculate wall thickness.
    # We assume the mask represents the pipe wall (ring).
    
    inner_cnt = None
    if len(contours) > 1:
        # Check if the second largest is actually inside the first (simple bounding box check or hierarchy)
        # For now, just take the second largest
        inner_cnt = contours[1]
    
    # Calculate Outer Geometry
    (x_out, y_out), radius_out = cv2.minEnclosingCircle(outer_cnt)
    metrics["diameter"] = radius_out * 2
    
    if inner_cnt is not None:
        (x_in, y_in), radius_in = cv2.minEnclosingCircle(inner_cnt)
        
        # Wall thickness (average)
        metrics["wall_thickness"] = radius_out - radius_in
        
        # Eccentricity: distance between centers
        center_dist = np.sqrt((x_out - x_in)**2 + (y_out - y_in)**2)
        metrics["eccentricity"] = center_dist
        
        metrics["valid"] = True
    else:
        # Fallback if only one contour found (e.g. solid disk or failed segmentation)
        metrics["valid"] = True # Still valid diameter
        metrics["wall_thickness"] = 0.0
        
    return metrics

def subpixel_edge_refinement(mask_prob):
    """
    Placeholder for sub-pixel edge refinement.
    """
    # This would implement the logic from res/DeepLabV3.py if needed
    pass
