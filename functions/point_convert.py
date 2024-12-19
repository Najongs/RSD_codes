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

# 객체의 HSV 범위로 마스크를 만들고, 중앙값 및 각도를 계산하는 함수
def get_object_center_and_angle(image, object_name):
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
        # 가장 큰 윤곽선을 찾아서 회전된 사각형으로 감싸기
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 사각형의 중앙값 계산 (회전된 사각형의 중심)
        center_x = int(rect[0][0])
        center_y = int(rect[0][1])

        # 사각형의 폭과 높이 추출
        width, height = rect[1]

        # 긴 변 기준 각도 계산
        if width < height:
            angle = rect[2]  # 세로가 더 길면 원래 각도 사용
        else:
            angle = rect[2] + 90  # 가로가 더 길면 90도 보정

        # 각도를 -90도 ~ 90도로 보정
        angle = angle % 180
        if angle > 90:
            angle -= 180

        # 원점 기준 좌표로 변환
        relative_x = (center_x - origin_x) * pixel_to_meter_ratio
        relative_y = (origin_y - center_y) * pixel_to_meter_ratio  # y축 반전

        # 결과 이미지 생성
        result_image = cv2.drawContours(image.copy(), [box], 0, (0, 255, 0), 2)

        # 바운딩 박스 왼쪽 위에 중앙 좌표 및 회전 각도 표시
        text = f"({relative_x:.2f}, {relative_y:.2f}), {angle:.2f} deg"
        cv2.putText(result_image, text, (center_x - 50, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 중심점 표시
        cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)

        # 리턴: 결과 이미지, 중앙 좌표, 회전 각도
        return  (relative_x, relative_y), angle

    return  (0, 0), 0
