import cv2  #图像处理OpenCV
import mediapipe as mp  #人体关键节点检测


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_point(landmarks, name):
    point = landmarks[mp_pose.PoseLandmark[name].value]
    return point.x, point.y, point.visibility


def recognize_action(landmarks):#y 越小，点越靠上。
    left_shoulder = get_point(landmarks, "LEFT_SHOULDER")
    right_shoulder = get_point(landmarks, "RIGHT_SHOULDER")
    left_wrist = get_point(landmarks, "LEFT_WRIST")
    right_wrist = get_point(landmarks, "RIGHT_WRIST")
    left_hip = get_point(landmarks, "LEFT_HIP")
    right_hip = get_point(landmarks, "RIGHT_HIP")
    left_knee = get_point(landmarks, "LEFT_KNEE")
    right_knee = get_point(landmarks, "RIGHT_KNEE")
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    wrist_y = (left_wrist[1] + right_wrist[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    knee_y = (left_knee[1] + right_knee[1]) / 2

    # 举手：两个手腕都高于肩膀
    if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
        return "Raise Two Hands"
    elif left_wrist[1] > left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
        return "Raise Left Hand"
    elif left_wrist[1] < left_shoulder[1] and right_wrist[1] > right_shoulder[1]:
        return "Raise Right Hand"
    # 下蹲
    # 站立时 hip_y 和 knee_y 差距较大；下蹲时髋部会下降，接近膝盖
    hip_knee_distance = knee_y - hip_y

    if hip_knee_distance < 0.18: #通过0.18可以调整下蹲精度
        return "Squat"

    return "Standing"

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败。可以把 VideoCapture(0) 改成 VideoCapture(1) 再试。")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("无法读取摄像头画面。")
        break

    frame = cv2.flip(frame, 1)#镜像翻转
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#转换成mediapipe读得懂的格式

    results = pose.process(rgb_frame)

    action = "No Person"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        action = recognize_action(landmarks)

        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.putText(
        frame,
        f"Action: {action}",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("My First Action Recognition Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()