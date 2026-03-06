
#[실시간 졸음 및 감정 감지 시스템을 통한 안전 운전 지원 시스템]
#파이썬 OpenCV 라이브러리인 CV2를 이용하여 실시간으로 카메라에서 나온 이미지를 캡처한 후 분석 도구인 FACE++ API와 dlib에 이미지를 전달한다. 
#FACE++ API는 운전자의 감정을 식별할 수 있는 도구로 사용되었는데 전달된 이미지 속 운전자의 Emotion이 ‘disgust’ 인지 여부를 확인하여 현재 감정 상태를 추정한다. 
#동시에, Dlib을 활용하여 전달된 이미지 속 운전자의 눈꺼풀이 일정 임계치 이상 감겨 있을 경우 졸음 상태로 인지한다.
#감정이 'disgust' 상태이거나 졸음 상태로 인지될 경우 온디바이스에서 실행되는 pyttsx3 및 Pygame을 통해 경고 메세지와 음악을 내보낸다.

import cv2
import dlib
import pyttsx3
import pygame
import time
import numpy as np
import requests

from scipy.spatial import distance

engine = pyttsx3.init()

pygame.mixer.init()

# Face++ API 설정하기
api_url = "https://api-us.faceplusplus.com/facepp/v3/detect"
api_key = "O0Yzg1_AztBGCA4zu1m40RPH3fayXv4k"
api_secret = "XCmCgjSttGBiMQc3ysTdl2aaDvJyhL0b"

# Dlib 얼굴 탐지기와 랜드마크 기능을 불러온다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 눈 깜빡임 감지를 위헤 EAR 임계값과 프레임 카운트를 설정한다.
EYE_AR_THRESHOLD = 0.2
EYE_AR_CONSEC_FRAMES = 3
closed_eye_frame_count = 0

# EAR 계산 함수
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 감정에 따른 음악 재생 및 음성 경고
def alert_speech(message):
    engine.say(message)
    engine.runAndWait()

def play_music(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(loops=0, start=0.0)
    except Exception as e:
        print(f"Error playing music: {e}")

def stop_music():
    pygame.mixer.music.stop()

# 비디오 캡처 설정
cap = cv2.VideoCapture(0)

# 졸음 상태를 감지한다.
is_drowsy = False
last_alert_time = 0  # 마지막 졸음 경고 시간
music_playing = False
is_disgust_alerted = False  # 감정 경고 상태 추적 변수
warning_flash_time = 0  # Warning 메시지 깜빡이기 시작한 시간

while True:
    ret, frame = cap.read()
    if not ret:
        break

 
    rgb_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _, img_encoded =cv2.imencode('.jpg', rgb_frame)
    img_data =img_encoded.tobytes()

    # Face++ API 요청하기
    data = {
        'api_key': api_key,
        'api_secret': api_secret,
        'return_attributes': 'emotion'
    }
    files = {'image_file': img_data}
    response =requests.post(api_url, data=data, files=files)
    result= response.json()

    # 감정 분석 후 감정 데이터를 추출한다.
    if 'faces' in result and len(result['faces'])>0:
        for face in result['faces']:
            if 'attributes' in face:
                emotions = face['attributes'].get('emotion', {})
                emotion = max(emotions, key=emotions.get)  # 가장 높은 감정 값을 가진 감정 추출
                emotion_text = f": {emotion}"

                # 감정에 따라 다른 색상 컬러를 나타낸다.
               
                if emotion == 'disgust':
                    emotion_text = "Irritation"
                    color = (0, 0, 255)  # 빨간색
                elif emotion == 'neutral':
                    emotion_text = "Natural"
                    color = (255, 255, 255)  # 흰색
                else:
                    continue

                # 화면에 감정 텍스트를 나타낸다.
                cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                if emotions.get('disgust', 0) >74:
                    if not is_disgust_alerted:
                        alert_speech("경고! 감정이 고조됐습니다. 편안하게 운전을 하는 것은 안전에 도움됩니다")
                        play_music("calm_music.mp3")
                        is_disgust_alerted = True
                    else:
                        # 음악이 이미 재생되고 있는 경우 경고 문구만 출력하고 음악 재생은 생략한다.
                        alert_speech("경고! 감정이 고조됐습니다.")
            if not pygame.mixer.music.get_busy():
                is_disgust_alerted = False  # 음악이 재생 된 이후, 감정이 낮아지면 초기화


    # dlib을 이용하여 눈 깜빡임 속도를 측정한다.
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray,face)

        # 왼쪽 /오른쪽 눈의 좌표 추출
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # 각 눈의 EAR을 계산
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # EAR이 임계값 이하이면 눈이 감긴 상태로 간주
        if ear < EYE_AR_THRESHOLD:
            closed_eye_frame_count += 1
            if closed_eye_frame_count >= EYE_AR_CONSEC_FRAMES and not is_drowsy:
                # 졸음 상태 처음 감지 시 경고 및 음악 시작
                alert_speech("경고!! 졸음운전이 감지됐습니다.")
                play_music("loud_alert_music.mp3")
                is_drowsy = True
                music_playing = True
                warning_flash_time = time.time()  # 경고 메시지 깜빡이기 시작 시간 기록
        else:
            closed_eye_frame_count = 0

        if ear >= EYE_AR_THRESHOLD:
            # 왼쪽 눈 감지 초록선
            cv2.polylines(frame, [cv2.convexHull(np.array(left_eye))], isClosed=True, color=(0, 255, 0), thickness=2)
            # 오른쪽 감지 초록선
            cv2.polylines(frame, [cv2.convexHull(np.array(right_eye))], isClosed=True, color=(0, 255, 0), thickness=2)


    # 음악이 재생 중이면 경고를 주기적으로 반복
    if music_playing and time.time() - last_alert_time > 4:
        alert_speech("경고!! 졸음운전이 감지됐습니다.")

    # 졸음 상태가 아니면 4초 후에 음악을 멈추고 마지막 경고 출력
    if closed_eye_frame_count < EYE_AR_CONSEC_FRAMES:
        if time.time() - last_alert_time > 4 and music_playing:
            stop_music()  # 음악 멈춤
            alert_speech("졸음운전은 위험합니다. 창문을 환기하고 인근의 졸음쉼터를 찾아 휴식하는 것을 권장합니다.")
            is_drowsy = False  # 졸음 상태 해제
            music_playing = False

    # 비디오 프레임
    cv2.imshow('Video Feed', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
