import cv2

video_path = "C001H002S0001.avi"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie udało się otworzyć filmu.")
else:
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frames / fps if fps > 0 else 0

    print(f"Liczba klatek: {frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Długość filmu: {duration:.2f} s")

cap.release()