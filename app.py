from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import threading
import time
import numpy as np

app = Flask(__name__)

# โหลดโมเดล
model = YOLO('best.pt')

# เปลี่ยนชื่อ label จาก mungbeans เป็น bean
model.names[0] = 'bean'
model.model.names[0] = 'bean'

latest_bean_count = 0
passed_bean_count = 0
line_y_percent = 50  # ตำแหน่งเส้นนับแนวนอน (% จากบน)
global_lock = threading.Lock()

# สถิติและประวัติสำหรับ Dashboard
start_time = time.time()
metrics_history = []  # เก็บ {time, passed, in_frame, ppm}
event_logs = []       # เก็บ {time, text, type}
last_history_time = 0

def add_event_log(text, log_type="info"):
    timestamp = time.strftime("%H:%M:%S")
    event_logs.append({"time": timestamp, "text": text, "type": log_type})
    if len(event_logs) > 30:
        event_logs.pop(0)

add_event_log("เริ่มต้นระบบตรวจจับและนับถั่วเขียว", "system")

# ระบบ tracking ที่ปรับปรุงแล้ว
# แต่ละ object เก็บ: cx, cy, counted, missing_frames, frames_above
tracked_objects = {}
next_obj_id = 0
MAX_MISSING = 15       # จำนวนเฟรมที่ให้ object หายได้ก่อนลบ (กันกะพริบ)
MIN_FRAMES_ABOVE = 3   # ต้องเห็นอยู่เหนือเส้นอย่างน้อย N เฟรม ก่อนจะนับข้ามเส้น
LINE_MARGIN = 15        # ระยะ hysteresis (px) กันสั่นไหวรอบเส้น


class VideoCamera(object):
    def __init__(self):
        self.video = None
        self.current_frame = None
        self.ret = False
        self.cam_id = 0
        self.running = True
        self.cam_lock = threading.Lock()
        
        if not self.start_camera(1):
            self.start_camera(0)
            
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def start_camera(self, cam_id):
        with self.cam_lock:
            if self.video is not None:
                self.video.release()
                self.video = None
                
            self.cam_id = cam_id
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[0] > 0:
                    self.video = cap
                    print(f"✅ เปิดกล้องสำเร็จที่ Index: {cam_id} (DSHOW)")
                    return True
            cap.release()
            print(f"❌ ไม่สามารถเปิดกล้องที่ Index: {cam_id}")
            return False

    def update(self):
        while self.running:
            with self.cam_lock:
                if self.video is not None:
                    ret, frame = self.video.read()
                    if ret:
                        self.ret, self.current_frame = ret, frame
            time.sleep(0.01)

    def get_frame(self):
        return self.ret, self.current_frame

    def __del__(self):
        self.running = False
        if self.video is not None:
            self.video.release()

camera = VideoCamera()


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """วาดเส้นประ"""
    x1, y1 = pt1
    x2, y2 = pt2
    total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if total_length == 0:
        return
    num_dashes = int(total_length / (dash_length * 2))
    for i in range(num_dashes + 1):
        start_ratio = (i * 2 * dash_length) / total_length
        end_ratio = min(((i * 2 + 1) * dash_length) / total_length, 1.0)
        sx = int(x1 + (x2 - x1) * start_ratio)
        sy = int(y1 + (y2 - y1) * start_ratio)
        ex = int(x1 + (x2 - x1) * end_ratio)
        ey = int(y1 + (y2 - y1) * end_ratio)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def match_detections(detections, tracked, max_dist=100):
    """จับคู่ detection กับ tracked objects ด้วยระยะทาง centroid"""
    matched = {}
    used_det = set()
    used_track = set()
    
    pairs = []
    for tid, tdata in tracked.items():
        for di, (cx, cy) in enumerate(detections):
            dist = np.sqrt((cx - tdata['cx']) ** 2 + (cy - tdata['cy']) ** 2)
            pairs.append((dist, tid, di))
    
    pairs.sort(key=lambda x: x[0])
    
    for dist, tid, di in pairs:
        if tid in used_track or di in used_det:
            continue
        if dist < max_dist:
            matched[tid] = di
            used_track.add(tid)
            used_det.add(di)
    
    return matched, used_det


def create_error_frame(text):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def generate_frames():
    global latest_bean_count, passed_bean_count, tracked_objects, next_obj_id
    
    error_frame = create_error_frame("No Camera Signal. Please click 'Switch Camera'.")
    
    while True:
        success, frame = camera.get_frame()
        if not success or frame is None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame + b'\r\n')
            time.sleep(0.1)
            continue
            
        img = frame.copy()
        h, w = img.shape[:2]
        
        with global_lock:
            ly = int(h * line_y_percent / 100)
        
        # รัน YOLO prediction
        results = model.predict(source=img, save=False, verbose=False, imgsz=480)
        
        boxes = results[0].boxes
        current_detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            current_detections.append((cx, cy))
        
        with global_lock:
            latest_bean_count = len(current_detections)
            
            matched, used_det = match_detections(current_detections, tracked_objects)
            new_tracked = {}
            
            # อัปเดต tracked objects ที่จับคู่ได้
            for tid, di in matched.items():
                cx, cy = current_detections[di]
                old_cy = tracked_objects[tid]['cy']
                counted = tracked_objects[tid]['counted']
                frames_above = tracked_objects[tid].get('frames_above', 0)
                
                # นับจำนวนเฟรมที่อยู่เหนือเส้น (ต้องเหนือพอสมควร - hysteresis)
                if cy < ly - LINE_MARGIN:
                    frames_above += 1
                elif cy > ly + LINE_MARGIN:
                    frames_above = 0  # ถ้าอยู่ใต้เส้นชัดเจน reset
                
                # นับเมื่อ: ยังไม่เคยนับ + เคยอยู่เหนือเส้นนานพอ + ตอนนี้ข้ามลงใต้เส้น
                if not counted and frames_above >= MIN_FRAMES_ABOVE and cy >= ly:
                    passed_bean_count += 1
                    counted = True
                    add_event_log(f"ถั่วเขียวข้ามเส้นสำเร็จ (รวม {passed_bean_count} เมล็ด)", "count")
                
                new_tracked[tid] = {
                    'cx': cx, 'cy': cy, 'counted': counted,
                    'missing': 0, 'frames_above': frames_above
                }
            
            # คง tracked objects ที่หายไปชั่วคราว (ไม่ถูกจับคู่)
            for tid, tdata in tracked_objects.items():
                if tid not in matched:
                    missing = tdata.get('missing', 0) + 1
                    if missing <= MAX_MISSING:
                        new_tracked[tid] = {
                            'cx': tdata['cx'], 'cy': tdata['cy'],
                            'counted': tdata['counted'],
                            'missing': missing,
                            'frames_above': tdata.get('frames_above', 0)
                        }
            
            # เพิ่ม detection ใหม่ที่ยังไม่ได้จับคู่
            for di, (cx, cy) in enumerate(current_detections):
                if di not in used_det:
                    # ถ้าเริ่มต้นอยู่ใต้เส้น ถือว่านับแล้ว (ไม่นับซ้ำ)
                    already_below = cy >= ly
                    new_tracked[next_obj_id] = {
                        'cx': cx, 'cy': cy, 'counted': already_below,
                        'missing': 0, 'frames_above': 0
                    }
                    next_obj_id += 1
            
            tracked_objects = new_tracked

            # อัปเดตสถิติรายวินาทีสำหรับ Dashboard
            now = time.time()
            if now - last_history_time >= 1.0:
                last_history_time = now
                elapsed = max(1.0, now - start_time)
                ppm = round((passed_bean_count / elapsed) * 60, 1)
                metrics_history.append({
                    "time": time.strftime("%H:%M:%S"),
                    "passed": passed_bean_count,
                    "in_frame": latest_bean_count,
                    "ppm": ppm
                })
                if len(metrics_history) > 40:
                    metrics_history.pop(0)
        
        # วาดผลลัพธ์
        res_plotted = results[0].plot(font_size=0.4, line_width=1)
        
        # วาดเส้นนับแนวนอน (สีแดง, เส้นทึบ)
        cv2.line(res_plotted, (0, ly), (w, ly), (0, 0, 255), 2)
        cv2.putText(res_plotted, "Counting Line", (10, ly - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # วาดเส้นประแนวตั้งกลางจอ (สีเหลืองอ่อน)
        mid_x = w // 2
        light_yellow = (153, 255, 255)  # BGR
        draw_dashed_line(res_plotted, (mid_x, 0), (mid_x, h), light_yellow, 1, 12)
        
        # แสดงจำนวนถั่วที่ผ่านเส้น
        with global_lock:
            count_text = f"Passed: {passed_bean_count}"
        cv2.putText(res_plotted, count_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', res_plotted)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_count')
def get_count():
    with global_lock:
        return jsonify({'count': latest_bean_count, 'passed': passed_bean_count})

@app.route('/get_dashboard_stats')
def get_dashboard_stats():
    with global_lock:
        elapsed = max(1.0, time.time() - start_time)
        ppm = round((passed_bean_count / elapsed) * 60, 1)
        return jsonify({
            'count': latest_bean_count,
            'passed': passed_bean_count,
            'ppm': ppm,
            'line_percent': line_y_percent,
            'cam_id': camera.cam_id,
            'uptime_seconds': int(elapsed),
            'history': metrics_history,
            'logs': list(reversed(event_logs))
        })

@app.route('/set_line/<int:percent>')
def set_line(percent):
    global line_y_percent
    percent = max(5, min(95, percent))
    with global_lock:
        line_y_percent = percent
        add_event_log(f"ปรับตำแหน่งเส้นนับเป็น {percent}%", "system")
    return jsonify({'success': True, 'percent': percent})

@app.route('/reset_count')
def reset_count():
    global passed_bean_count, tracked_objects, next_obj_id, start_time, metrics_history
    with global_lock:
        passed_bean_count = 0
        tracked_objects = {}
        next_obj_id = 0
        start_time = time.time()
        metrics_history.clear()
        add_event_log("รีเซ็ตระบบการนับและสถิติเรียบร้อย", "warning")
    return jsonify({'success': True})

@app.route('/switch_camera/<int:cam_id>')
def switch_camera(cam_id):
    success = camera.start_camera(cam_id)
    with global_lock:
        if success:
            add_event_log(f"สลับไปใช้กล้อง Index {cam_id} สำเร็จ", "system")
        else:
            add_event_log(f"ไม่พบสัญญาณกล้อง Index {cam_id}", "error")
    return jsonify({'success': success, 'cam_id': cam_id})

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
