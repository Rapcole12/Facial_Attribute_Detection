
import torchvision.ops

def extract_patches(images_normalized, bboxes_batch, expected_size=(24, 24)):

    # Get the shape of the input images
    batch_size, _, img_height, img_width = images_normalized.shape

    selector = [1, 0, 3, 2]  # Correct the order of bounding box coordinates
    bboxes_batch = bboxes_batch[:, selector]

    rois = []
    for i in range(batch_size):
        rois.append(bboxes_batch[i].unsqueeze(0))

    # Perform ROI Align for cropping and resizing
    patches = torchvision.ops.roi_align(
        images_normalized,            # Input image tensor
        rois,                       # Regions of Interest (ROIs)
        output_size=expected_size,    # Target size (height, width)
        spatial_scale=1.0,            # Since coordinates are normalized
        sampling_ratio=-1             # Automatic adaptive sampling
    )
    
    return patches