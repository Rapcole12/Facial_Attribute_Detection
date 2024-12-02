
import torchvision.ops
import numpy as np
import torch

def extract_patches(images_normalized, bboxes_batch, expected_size=(24, 24)):
    # Get the shape of the input images
    batch_size, _, img_height, img_width = images_normalized.shape

    selector = [1, 0, 3, 2]  # Correct the order of bounding box coordinates
    bboxes_batch = bboxes_batch[:, selector]

    # Ensure bboxes_batch is a torch tensor
    if isinstance(bboxes_batch, np.ndarray):
        bboxes_batch = torch.tensor(bboxes_batch, dtype=torch.float32)

    device = images_normalized.device  # This assumes images_normalized is already on the correct device (CPU or CUDA)

    # Initialize a list for ROIs
    rois = []
    for i in range(batch_size):
        # Add unsqueeze to make the bbox a 1x4 tensor
        rois.append(bboxes_batch[i].unsqueeze(0).to(device))  # Move ROIs to the same device as the images 

    # Perform ROI Align for cropping and resizing
    patches = torchvision.ops.roi_align(
        images_normalized,            # Input image tensor
        rois,                         # Regions of Interest (ROIs)
        output_size=expected_size,    # Target size (height, width)
        spatial_scale=1.0,            # Since coordinates are normalized
        sampling_ratio=-1             # Automatic adaptive sampling
    )
    
    return patches

def generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides=2, cell_size=12):
    """
    Generates bounding boxes for detected objects (e.g., faces) based on the class and regression outputs of a model,
    supporting batch input.
    
    Args:
        bbox_reg (tf.Tensor): Bounding box regression predictions with shape (batch_size, height, width, 4).
                              This contains adjustments to apply to the initial bounding box positions for each image in the batch.
        bbox_class (tf.Tensor): Class predictions (e.g., face/non-face) of shape (batch_size, height, width, 2),
                                where the second channel corresponds to the probability of a face being present.
        threshold_face (float): A threshold between 0 and 1 that determines if a detection is considered a face or not.
                                Bounding boxes are only generated for detections with probabilities greater than this value.
        strides (int, optional): The step size (in pixels) used to slide the detection window over the image. Default is 2.
        cell_size (int, optional): The size of the sliding window (in pixels) used to detect faces. Default is 12.
    
    Returns:
        np.ndarray: An array of bounding boxes for the entire batch, where each box is represented as 
                    [batch_index, x1, y1, x2, y2, confidence].
                    The `batch_index` indicates which image in the batch the bounding box belongs to.
    """
    bbox_reg = bbox_reg.cpu().detach().numpy()
    bbox_class = bbox_class.cpu().detach().numpy()

    bbox_class = np.transpose(bbox_class, (0, 2, 3, 1))
    bbox_reg = np.transpose(bbox_reg, (0, 2, 3, 1))

    # Create a mask for detected faces based on the threshold for face probability
    confidence_score = bbox_class[:,:,:,1]

    # Find the indices where the detection mask is true (i.e., face detected)
    index_bboxes = np.stack(np.where(confidence_score > threshold_face)) # batch_size, y, x
    filtered_bbox_reg = np.transpose(bbox_reg[index_bboxes[0], index_bboxes[1], index_bboxes[2]], (1,0))

    # Extract the regression values
    reg_x1, reg_y1, reg_x2, reg_y2 = filtered_bbox_reg

    # Convert strides and cell size into arrays for easy broadcasting
    strides = np.asarray([[1], [strides], [strides]])
    cellsize = [np.asarray([[0], [1], [1]]), np.asarray([[0], [cell_size], [cell_size]])]

    # Calculate the top-left and bottom-right corners of the bounding boxes
    bbox_up_left = index_bboxes * strides + cellsize[0]
    bbox_bottom_right = index_bboxes * strides + cellsize[1]

    # Calculate width and height for the bounding boxes
    reg_w = bbox_bottom_right[2] - bbox_up_left[2]  # width of bounding box
    reg_h = bbox_bottom_right[1] - bbox_up_left[1]  # height of bounding box

    # Apply the regression to adjust the bounding box coordinates
    x1 = bbox_up_left[2] + reg_x1 * reg_w  # Adjusted x1
    y1 = bbox_up_left[1] + reg_y1 * reg_h  # Adjusted y1
    x2 = bbox_bottom_right[2] + reg_x2 * reg_w  # Adjusted x2
    y2 = bbox_bottom_right[1] + reg_y2 * reg_h  # Adjusted y2

    # Concatenate the bounding box coordinates and detection information, keeping batch index
    bboxes_result = np.stack([x1, y1, x2, y2], axis=0).T

    return bboxes_result

def replace_confidence(bboxes_batch, new_scores):
    """
    Replaces the confidence scores of bounding boxes with new scores provided.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row
                                   contains the bounding box coordinates and the confidence score.
                                   The confidence score is expected to be in the last column.
        new_scores (np.ndarray): An array of new confidence scores of shape (n, m), where the 
                                 confidence score is also expected to be in the last column.

    Returns:
        np.ndarray: The bounding boxes array with updated confidence scores from `new_scores`.
    """
    bboxes_batch[:, -1] = new_scores[:, -1]
    return bboxes_batch

def adjust_bboxes(bboxes_batch, bboxes_offsets):
    """
    Adjusts the bounding box coordinates by applying the provided offsets.
    
    The offsets are applied to resize and shift the bounding boxes based on their width and height.

    Args:
        bboxes_batch (np.ndarray): An array of bounding boxes of shape (n, m), where each row contains
                                   the batch index, bounding box coordinates [x1, y1, x2, y2], and
                                   potentially additional data such as scores.
        bboxes_offsets (np.ndarray): An array of offsets for adjusting the bounding boxes. The shape should be
                                     (n, 4), where each row contains offsets for [dx1, dy1, dx2, dy2].

    Returns:
        np.ndarray: The adjusted bounding boxes with updated coordinates, maintaining any additional columns 
                    beyond the bounding box coordinates (such as scores).
    """
    bboxes_batch = bboxes_batch.clone()  # Use clone() instead of copy()
    
    # Ensure that tensors are on CPU if necessary
    bboxes_batch = bboxes_batch.cpu()
    bboxes_offsets = bboxes_offsets.cpu()

    # Calculate width and height of each bounding box
    w = bboxes_batch[:, 2] - bboxes_batch[:, 0] + 1
    h = bboxes_batch[:, 3] - bboxes_batch[:, 1] + 1

    # Ensure that sizes have the shape (n, 4) to match bboxes_offsets
    sizes = torch.stack([w, h, w, h], dim=-1)  # (n, 4), width and height for each bounding box

    # Apply offsets to the bounding box coordinates
    bboxes_batch[:, 0:4] += bboxes_offsets * sizes

    return bboxes_batch

def resize_to_square(bboxes):
    """
    Adjusts bounding boxes to be square by resizing them based on their largest dimension 
    (width or height). The bounding boxes are resized by expanding the smaller dimension
    to match the larger one while keeping the center of the box intact.

    Args:
        bboxes (np.ndarray): An array of bounding boxes of shape (n, 5), where each row
                             represents [batch_index, x1, y1, x2, y2].

    Returns:
        np.ndarray: An array of bounding boxes adjusted to be square, maintaining their center positions.
    """
    bboxes = bboxes.copy()

    h = bboxes[:, 3] - bboxes[:, 1]  # Height of each bounding box
    w = bboxes[:, 2] - bboxes[:, 0]  # Width of each bounding box
    largest_size = np.maximum(w, h)  # Largest dimension (width or height)

    # Adjust x1 and y1 to center the bounding box and resize to square
    bboxes[:, 0] = np.abs(bboxes[:, 0] + w * 0.5 - largest_size * 0.5)
    bboxes[:, 1] = np.abs(bboxes[:, 1] + h * 0.5 - largest_size * 0.5)
    bboxes[:, 2:4] = np.abs(bboxes[:, 0:2] + np.tile(largest_size, (2, 1)).T)  # Resize x2, y2

    return bboxes

def generate_bounding_box_after_pnet(bbox_reg, bbox_class, threshold_face, strides=2, cell_size=12):

    # Reshape the inputs to expected shapes
    # Ensure tensors are on the CPU for NumPy compatibility
    bbox_reg = bbox_reg.cpu().detach().numpy().reshape((-1, 4))  # (batch_size * height * width, 4)
    bbox_class = bbox_class.cpu().detach().numpy().reshape((-1, 2))  # (batch_size * height * width, 2)

    # Extract confidence scores
    confidence_score = bbox_class[:, 1]

    # Filter detections based on threshold
    valid_indices = np.where(confidence_score > threshold_face)[0]
    if valid_indices.size == 0:
        return np.empty((0, 4)), np.empty(0)  # Return empty arrays if no detections meet the threshold

    filtered_bbox_reg = bbox_reg[valid_indices]
    confidence_score = confidence_score[valid_indices]

    # Calculate bounding box coordinates
    x1 = np.abs(((valid_indices % strides) * strides) + (filtered_bbox_reg[:, 0] * cell_size).astype(int))
    y1 = np.abs(((valid_indices // strides) * strides) + (filtered_bbox_reg[:, 1] * cell_size).astype(int))
    x2 = np.abs(x1 + (filtered_bbox_reg[:, 2] * cell_size).astype(int))
    y2 = np.abs(y1 + (filtered_bbox_reg[:, 3] * cell_size).astype(int))

    # Combine into a bounding box array
    bboxes_result = np.stack([x1, y1, x2, y2], axis=-1)

    return bboxes_result