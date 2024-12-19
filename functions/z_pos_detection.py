import cv2
import numpy as np

# Object의 HSV 범위 설정 (예시)
object_hsv_ranges = {
    "cuboid": (0, 6, 131, 255, 32, 233),
    "hexagonal_prism": (85, 111, 202, 255, 67, 242),
    "needle": (127, 173, 46, 255, 139, 255),
    "torus": (9, 48, 252, 255, 1, 255),
    "tube": (46, 88, 134, 255, 75, 255),
    "cylinder": (114, 139, 173, 255, 75, 255)
}

def process_image_and_depth(image_path, depth_image_path, object_name, scaling_factor):
    def get_object_center_and_angle(image, object_name):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_min, h_max, s_min, s_max, v_min, v_max = object_hsv_ranges.get(object_name, (0, 0, 0, 0, 0, 0))
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            center_x = int(rect[0][0])
            center_y = int(rect[0][1])
            return (center_x, center_y)
        else:
            return (0, 0)

    def read_depth_image(filepath):
        depth_image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise FileNotFoundError(f"File not found: {filepath}")
        return depth_image

    # Load RGB image
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image_path
    if image is None:
        raise FileNotFoundError(f"RGB Image not found at {image_path}")

    # Load Depth image and process
    # depth_image = read_depth_image(depth_image_path)
    depth_image = depth_image_path
    z_values = depth_image.astype(np.float32) * scaling_factor

    # Get object center pixel
    center_pixel = get_object_center_and_angle(image, object_name)
    pixel_x, pixel_y = int(round(center_pixel[0])), int(round(center_pixel[1]))

    # Get Z value at the object's center pixel
    z_value = z_values[pixel_y, pixel_x]
    return center_pixel, z_value


# Example Usage
if __name__ == "__main__":
    # 파일 경로 및 파라미터 설정
    depth_image_path = "/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka/images/Test_depth.png"
    image_path = "/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka/images/Test_rgb.png"
    object_list = ["cylinder", "hexagonal_prism", "needle", "torus", "tube", "cuboid"]
    scaling_list = [0.0004, 0.00043396, 0.0004, 0.0004, 0.00043333, 0.00039474]
    object_name = object_list[0]
    scaling_factor = scaling_list[0]
    
    center_pixel, z_value = process_image_and_depth(image_path, depth_image_path, object_name, scaling_factor)
    print(z_value)
