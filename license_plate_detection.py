import cv2
import re
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import threading
import mysql.connector
import time
from ultralytics import YOLO
import easyocr
from fuzzywuzzy import process, fuzz

# --- Database Connection Settings ---
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = ''
DB_NAME = 'license_plates_db'

# --- ฟังก์ชันเชื่อมต่อ MySQL ---
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4'
        )
        return conn
    except mysql.connector.Error as err:
        print(f"❌ Error connecting to database: {err}")
        return None

# --- ฟังก์ชันตรวจสอบป้ายทะเบียน (แก้ไขใหม่) ---
def check_license_plate(plate_number, province):
    """ ตรวจสอบว่าป้ายทะเบียนและจังหวัดอยู่ใน MySQL โดยต้องอยู่ในแถวเดียวกัน """
    conn = get_db_connection()
    if conn is None:
        return False

    cursor = conn.cursor(buffered=True)
    try:
        cleaned_plate = clean_ocr_text(plate_number)
        cleaned_province = clean_ocr_text(province)

        print(f"🔍 OCR อ่านได้: '{plate_number}' / '{province}'")
        print(f"🔍 ค้นหาในฐานข้อมูล: '{cleaned_plate}' / '{cleaned_province}'")

        # ✅ ค้นหาข้อมูลจากฐานข้อมูล
        cursor.execute("SELECT plate_number, province FROM license_plates")
        all_db_data = cursor.fetchall()

        # ✅ ค้นหา plate_number ที่ใกล้เคียงที่สุด
        best_match_plate, plate_score = process.extractOne(cleaned_plate, [db_plate for db_plate, _ in all_db_data], scorer=fuzz.ratio)

        # ✅ ดึง province ที่ตรงกับป้ายทะเบียนที่เจอ
        cursor.execute("SELECT province FROM license_plates WHERE plate_number = %s", (best_match_plate,))
        matched_province = cursor.fetchone()

        if matched_province:
            matched_province = matched_province[0]
            province_score = fuzz.ratio(cleaned_province, matched_province)

            print(f"🔎 ค่าความคล้ายกัน: ป้ายทะเบียน={plate_score}%, จังหวัด={province_score}%")
            print(f"📌 ตรวจสอบกับฐานข้อมูล: '{best_match_plate}' / '{matched_province}'")

            # ✅ ป้ายทะเบียนและจังหวัดต้องตรงกัน (ใช้ค่าความคล้ายกัน > 85%)
            if plate_score >= 85 and province_score >= 70:
                print(f"✅ พบข้อมูลที่ตรงกันในฐานข้อมูล")
                return True

        print(f"❌ ไม่พบข้อมูลที่ตรงกันในฐานข้อมูล!")
        return False
    except mysql.connector.Error as err:
        print(f"⚠️ Database query error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()

# --- OCR Cleaning Function ---
def clean_ocr_text(text):
    """ ทำความสะอาดข้อความ OCR """
    text = text.upper().strip()
    text = re.sub(r'\s+', '', text)  # ลบช่องว่างทั้งหมด
    text = re.sub(r'[^0-9ก-๙]', '', text)  # คงไว้เฉพาะตัวเลขและอักษรไทย
    return text

# --- Settings ---
rtsp_url = "rtsp://Ballop:ballopop@172.20.10.2:554/stream1"
FONT_PATH = 'fonts/KanitBold.ttf'
font_size = 30
pil_font = ImageFont.truetype(FONT_PATH, font_size)

# --- YOLO Model ---
model = YOLO("lp_detector.pt")

# --- EasyOCR Reader ---
reader = easyocr.Reader(['th', 'en'])
reader = easyocr.Reader(['th', 'en'])
reader = easyocr.Reader(['th', 'en'])

# --- Video Stream Class ---
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

# --- วาดข้อความพร้อมพื้นหลังโปร่งใส ---
def put_text_with_pil(img, text, position, font, text_color, bg_color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil, "RGBA")

    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]

    x, y = position
    draw.rectangle([x - 5, y - 5, x + text_width + 15, y + text_height + 10], fill=bg_color)
    draw.text(position, text, font=font, fill=text_color)

    return np.array(img_pil)

# --- Main Processing ---
video_stream = VideoStream(rtsp_url)

while True:
    frame = video_stream.read()
    if frame is None:
        continue

    frame = cv2.resize(frame, (1280, 720))
    detections = []

    try:
        results = model(frame)
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]

                roi = frame[y1:y2, x1:x2]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ocr_results = reader.readtext(roi_gray)

                if len(ocr_results) >= 3:
                    plate_text = clean_ocr_text(ocr_results[0][1] + ocr_results[2][1])
                    province_text = clean_ocr_text(ocr_results[1][1])

                    detections.append((plate_text, province_text, x1, y1, x2, y2))

    except Exception as e:
        print(f"⚠️ Error: {e}")

    for plate_text, province_text, x1, y1, x2, y2 in detections:
        found = check_license_plate(plate_text, province_text)

        if found:
            text_display = f"{plate_text} ({province_text}) - พบในฐานข้อมูล ✅"
            text_color = (0, 255, 0)  
            bg_color = (0, 128, 0, 180)  
        else:
            text_display = f"{plate_text} ({province_text}) - ไม่พบในฐานข้อมูล ❌"
            text_color = (255, 0, 0)  
            bg_color = (128, 0, 0, 180)  

        cv2.rectangle(frame, (x1, y1), (x2, y2), text_color, 3)
        frame = put_text_with_pil(frame, text_display, (x1, max(30, y1 - 80)), pil_font, text_color, bg_color)

    cv2.imshow('License Plate Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_stream.stop()
cv2.destroyAllWindows()
