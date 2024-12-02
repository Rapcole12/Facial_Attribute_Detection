import numpy as np
from your_module_name import generate_bounding_box  # Replace with the actual module name

def test_generate_bounding_box():
    # Test case 1: Valid inputs with high confidence scores
    bbox_reg = np.array([
        0.1, 0.1, 0.2, 0.2,  # Bounding box 1 (regression values)
        0.3, 0.3, 0.4, 0.4   # Bounding box 2 (regression values)
    ])
    bbox_class = np.array([
        0.1, 0.9,  # Bounding box 1 (low, high confidence)
        0.2, 0.8   # Bounding box 2 (low, high confidence)
    ])
    threshold_face = 0.5
    strides = 2
    cell_size = 12

    result = generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides, cell_size)
    print("Test Case 1 - Valid Inputs:")
    print(result)

    # Test case 2: Confidence scores below the threshold
    bbox_reg = np.array([
        0.1, 0.1, 0.2, 0.2,
        0.3, 0.3, 0.4, 0.4
    ])
    bbox_class = np.array([
        0.1, 0.4,  # Bounding box 1 (below threshold)
        0.2, 0.3   # Bounding box 2 (below threshold)
    ])

    result = generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides, cell_size)
    print("\nTest Case 2 - Confidence Below Threshold:")
    print(result)

    # Test case 3: Empty input
    bbox_reg = np.array([])
    bbox_class = np.array([])

    result = generate_bounding_box(bbox_reg, bbox_class, threshold_face, strides, cell_size)
    print("\nTest Case 3 - Empty Input:")
    print(result)

    # Add more test cases as needed to test edge cases or boundary conditions.

if __name__ == "__main__":
    test_generate_bounding_box()