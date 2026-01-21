import numpy as np
import tensorflow as tf

def horizontal_flip(image):
    """
    Flip image horizontally (left-right).
    
    Args:
        image: Input image array
        
    Returns:
        Horizontally flipped image
    """
    return image[:, ::-1, :]

def vertical_flip(image):
    """
    Flip image vertically (up-down).
    
    Args:
        image: Input image array
        
    Returns:
        Vertically flipped image
    """
    return image[::-1, :, :]

def tta_predict(model, image, threshold=0.5):
    """
    Test Time Augmentation (TTA) for polyp segmentation.
    
    Applies horizontal and vertical flips to the input image,
    generates predictions for each augmentation, then averages
    the results for improved segmentation accuracy.
    
    Args:
        model: Trained IRv2-Net model
        image: Input image (H, W, C) - should be normalized
        threshold: Threshold for binary mask (default: 0.5)
        
    Returns:
        mean_mask: Averaged prediction mask (H, W, 1)
        binary_mask: Thresholded binary mask (H, W, 1)
    """
    # Original image
    n_image = image
    
    # Augmented versions
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)
    
    # Predict on all versions
    n_mask = model.predict(np.expand_dims(n_image, axis=0), verbose=0)[0]
    h_mask = model.predict(np.expand_dims(h_image, axis=0), verbose=0)[0]
    v_mask = model.predict(np.expand_dims(v_image, axis=0), verbose=0)[0]
    
    # Flip predictions back to original orientation
    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)
    
    # Average all predictions
    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    
    # Apply threshold for binary mask
    binary_mask = (mean_mask > threshold).astype(np.float32)
    
    return mean_mask, binary_mask

def tta_predict_batch(model, images, threshold=0.5):
    """
    Batch TTA prediction for multiple images.
    
    Args:
        model: Trained IRv2-Net model
        images: Batch of images (N, H, W, C)
        threshold: Threshold for binary mask (default: 0.5)
        
    Returns:
        mean_masks: Averaged prediction masks (N, H, W, 1)
        binary_masks: Thresholded binary masks (N, H, W, 1)
    """
    mean_masks = []
    binary_masks = []
    
    for image in images:
        mean_mask, binary_mask = tta_predict(model, image, threshold)
        mean_masks.append(mean_mask)
        binary_masks.append(binary_mask)
    
    return np.array(mean_masks), np.array(binary_masks)

def tta_model(model, image):
    """
    Original TTA function for backward compatibility.
    
    Args:
        model: Trained IRv2-Net model
        image: Input image (H, W, C)
        
    Returns:
        mean_mask: Averaged prediction mask
    """
    n_image = image
    h_image = horizontal_flip(image)
    v_image = vertical_flip(image)

    n_mask = model.predict(np.expand_dims(n_image, axis=0), verbose=0)[0]
    h_mask = model.predict(np.expand_dims(h_image, axis=0), verbose=0)[0]
    v_mask = model.predict(np.expand_dims(v_image, axis=0), verbose=0)[0]

    h_mask = horizontal_flip(h_mask)
    v_mask = vertical_flip(v_mask)

    mean_mask = (n_mask + h_mask + v_mask) / 3.0
    return mean_mask