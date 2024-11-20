import cv2
import face_recognition
import time
from playsound import playsound
import os

# โหลดรูปอ้างอิงและตรวจสอบไฟล์
try:
    reference_image = face_recognition.load_image_file("reference.jpg")
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
except (FileNotFoundError, IndexError):
    print("ไฟล์ reference.jpg ไม่ถูกต้องหรือไม่พบ!")
    exit(1)

if not os.path.exists("success.mp3"):
    print("ไฟล์เสียง success.mp3 ไม่พบ!")
    exit(1)

# เริ่มต้นการใช้กล้อง
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ตัวแปรสำหรับจัดการเสียงและการจับคู่
last_sound_time = 0
last_detection_time = time.time()
cooldown = 5  # วินาที
idle_time = 15  # วินาที
match_found = False  # ตัวแปรสถานะ

# ฟังก์ชันสำหรับดึงตำแหน่งและ encodings ของใบหน้า
def get_face_encodings(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small_frame)
    encodings = face_recognition.face_encodings(rgb_small_frame, locations)
    return locations, encodings

# วนลูปการทำงานของกล้อง
while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถเปิดกล้องได้")
        break

    # ตรวจจับใบหน้าและ encoding
    face_locations, face_encodings = get_face_encodings(frame)

    if not face_locations:
        # แสดงเฟรมปกติ (ไม่มีการจับใบหน้า)
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ถ้าตรวจจับไม่พบใบหน้าเป็นเวลา 15 วิ ให้บังคับหยุดการทำงานของโปรแกรมทันที
        if time.time() - last_detection_time > idle_time:
            print("ไม่มีการตรวจจับใบหน้าเป็นเวลา 15 วินาที หยุดโปรแกรม")
            break


        cv2.imshow("Face Comparison", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # อัปเดตเวลา ณ ปัจจุบันที่ตรวจพบใบหน้า
    last_detection_time = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # คำนวณระยะห่าง (distance)
        distance = face_recognition.face_distance([reference_encoding], face_encoding)[0]

        # ขยายตำแหน่งใบหน้ากลับเป็นขนาดภาพจริง
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # กำหนดสีและข้อความตามเงื่อนไข
        if distance < 0.4:  # หน้าที่ใกล้เคียงมากที่สุด (ตรงกับ reference)
            color = (255, 0, 0)  # สีน้ำเงิน
            label = f"Perfect Match (Distance: {distance:.2f})"
            if not match_found:  # เล่นเสียงเฉพาะเมื่อยังไม่เล่นในรอบนี้
                playsound("success.mp3")
                match_found = True
        elif distance < 0.6:  # หน้าแค่ใกล้เคียง
            color = (0, 255, 0)  # สีเขียว
            label = f"Close Match (Distance: {distance:.2f})"
            match_found = False  # รีเซ็ตสถานะ
        else:  # ไม่ตรง หรือ ไม่ใกล้เคียงเลย
            color = (0, 0, 255)  # สีแดง
            label = f"No Match (Distance: {distance:.2f})"
            match_found = False  # รีเซ็ตสถานะ

        # วาดกรอบและข้อความ
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # เพิ่มวันที่และเวลาบนหน้าจอ
    cv2.putText(frame, time.strftime("%Y-%m-%d %H:%M:%S"), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # แสดงภาพที่หน้าจอ
    cv2.imshow("Face Comparison", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดการใช้งานกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
