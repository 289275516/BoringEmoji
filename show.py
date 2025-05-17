import cv2
from ultralytics import YOLO
import numpy as np

# 加载 YOLOv8 模型
model = YOLO("best.pt")
model.model.names = {
    0: '愤怒',
    1: '蔑视',
    2: '恶心',
    3: '恐惧',
    4: '快乐',
    5: '中性',
    6: '悲伤',
    7: '惊喜'
}

# Load the images with alpha channel
happy_img = cv2.imread("Happy.png", cv2.IMREAD_UNCHANGED)
if happy_img is None:
    print("Error: Could not load Happy.png")
    exit()

anger_img = cv2.imread("Anger.png", cv2.IMREAD_UNCHANGED)
if anger_img is None:
    print("Error: Could not load Anger.png")
    exit()

disgust_img = cv2.imread("Disgust.png", cv2.IMREAD_UNCHANGED)
if disgust_img is None:
    print("Error: Could not load Disgust.png")
    exit()

# 获取摄像头内容，windows系统参数写0就行，mac最近有个连手机摄像头的默认功能，我关不掉，你如果也有这个问题参数就写1
cap = cv2.VideoCapture(1)

# 设置显示窗口的期望大小
DISPLAY_WIDTH = 300  # 可以根据需要调整这个值

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    # 分离通道
    if overlay.shape[2] == 4:  # 如果有alpha通道
        alpha = overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        rgb = overlay[:, :, :3]
        
        # 确保坐标在有效范围内
        if y + h > background.shape[0] or x + w > background.shape[1]:
            return background
            
        # 获取背景区域
        bg_region = background[y:y+h, x:x+w]
        
        # 混合图像
        blended = bg_region * (1 - alpha) + rgb * alpha
        background[y:y+h, x:x+w] = blended.astype(np.uint8)
    
    return background

def resize_frame(frame, target_width):
    # 计算缩放比例
    height, width = frame.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    # 使用INTER_AREA插值方法进行缩放，这种方法对于缩小图像效果最好
    return cv2.resize(frame, (target_width, new_height), interpolation=cv2.INTER_AREA)

while True:
    success, frame = cap.read()
    if success:
        # 调整检测框的坐标以匹配缩放后的图像
        scale = DISPLAY_WIDTH / frame.shape[1]
        
        results = model(source=frame, show=False, conf=0.8)

        for result in results:
            boxes = result.boxes
            
            # 只处理第一个检测到的人脸
            if len(boxes) > 0:
                box = boxes[0]  # 获取第一个检测框
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name
                class_name = result.names[cls]
                
                # If the detected class is "快乐"
                if class_name == "快乐":
                    # Resize happy image to fit the box
                    box_width = x2 - x1
                    box_height = y2 - y1
                    happy_resized = cv2.resize(happy_img, (box_width, box_height))
                    frame = overlay_transparent(frame, happy_resized, x1, y1)
        
        # 缩放显示画面
        frame = resize_frame(frame, DISPLAY_WIDTH)
        
        # Display the frame
        cv2.imshow("YOLO Detection", frame)

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

