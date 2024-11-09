#Para entrenar
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=5, imgsz=512, batch=2, device='cpu') 