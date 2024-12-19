import cv2
import numpy as np

# 기준값 설정
image_width, image_height = 1280, 720
origin_x, origin_y = 639, 665  # (0, 0) m가 대응하는 픽셀 좌표
pixel_to_meter_ratio = 0.5 / 307  # 307픽셀 = 0.5m 이므로 1픽셀당 몇 미터인지 계산

# Object의 HSV 범위 설정 (예시)
object_hsv_ranges = {
    "cuboid": (0, 6, 131, 255, 32, 233),
    "hexagonal_prism": (85, 111, 202, 255, 67, 242),
    "needle": (127, 173, 46, 255, 139, 255),
    "torus": (9, 48, 252, 255, 1, 255),
    "tube": (46, 88, 134, 255, 75, 255),
    "cylinder": (114, 139, 173, 255, 75, 255)
}

# 객체의 HSV 범위로 마스크를 만들고, 중앙값 계산 함수
def get_object_center(image, object_name):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Object의 HSV 범위 가져오기
    h_min, h_max, s_min, s_max, v_min, v_max = object_hsv_ranges.get(object_name, (0, 0, 0, 0, 0, 0))
    
    # HSV 범위로 마스크 생성
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # 마스크 이미지에서 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 가장 큰 윤곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)

        # 중심 좌표 계산
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])

        # 원점 기준 좌표 변환
        relative_x = (center_x - origin_x) * pixel_to_meter_ratio
        relative_y = (origin_y - center_y) * pixel_to_meter_ratio  # y축 반전

        return np.array([relative_y, relative_x])

    return np.array([0,0])

# 객체 간 겹침 확인 및 이동 방향 결정
def resolve_overlap_with_direction(object_positions, min_dist_threshold=0.15, move_distance=0.05):
    new_positions = np.array(object_positions)
    for i in range(len(new_positions)):
        for j in range(i + 1, len(new_positions)):
            dist_x = abs(new_positions[j][0] - new_positions[i][0])
            dist_y = abs(new_positions[j][1] - new_positions[i][1])
            
            # x 또는 y 방향으로 겹치는 경우
            if dist_x < min_dist_threshold and dist_y < min_dist_threshold:
                # 이동 방향 결정: 더 여유 있는 축으로 이동
                if dist_x < dist_y:
                    if new_positions[i][0] < new_positions[j][0]:
                        new_positions[i][0] -= move_distance
                        new_positions[j][0] += move_distance
                    else:
                        new_positions[i][0] += move_distance
                        new_positions[j][0] -= move_distance
                else:
                    if new_positions[i][1] < new_positions[j][1]:
                        new_positions[i][1] -= move_distance
                        new_positions[j][1] += move_distance
                    else:
                        new_positions[i][1] += move_distance
                        new_positions[j][1] -= move_distance

    return new_positions

# 메인 실행
if __name__ == "__main__":
    # 테스트 이미지 로드
    image_path = "/home/ryan/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/api/omni.isaac.franka/images/Test_rgb.png"
    image = cv2.imread(image_path)

    object_positions = []
    object_names = list(object_hsv_ranges.keys())
    
    # 객체 위치 추출
    for obj_name in object_names:
        position = get_object_center(image, obj_name)
        if position is not None:  # 유효한 위치만 추가
            object_positions.append(position)
            print(f"Object: {obj_name}, Position: {position}")

    # 겹침 해결
    if len(object_positions) > 1:
        adjusted_positions, coordinate = resolve_overlap_with_direction(object_positions, min_dist_threshold=0.15, move_distance=0.05)
        for i, adjusted_pos in enumerate(adjusted_positions):
            if not np.array_equal(object_positions[i], adjusted_pos):  # 이동한 경우만 출력
                print(f"Object {object_names[i]} moved to {adjusted_pos}, {coordinate}")
