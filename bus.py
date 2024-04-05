from ultralytics import YOLO
import cv2
import math
import pytesseract

# Kamera ayarları
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Genişlik
cap.set(4, 480)  # Yükseklik

# YOLO modelini yükleme
model = YOLO("yolo-Weights/yolov8n.pt")

# Sınıf isimleri (COCO dataset)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]  # Yukarıda verilen sınıf isimlerinin listesi

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # int değerlerine çevir

            cls_idx = int(box.cls[0])

            if classNames[cls_idx] == "bus":
                # Otobüs etrafına dikdörtgen çiz
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0]*100))/100

                # Tespit edilen otobüs bölgesini kes
                bus_image = img[y1:y2, x1:x2]

                # OCR ile otobüs numarasını oku
                bus_number = pytesseract.image_to_string(bus_image, config='--psm 6')
                print("Tespit edilen otobüs numarası:", bus_number)

                # Otobüs numarasını görüntüye yaz
                cv2.putText(img, bus_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Görüntüyü göster
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
