from flask import Flask, render_template, request, jsonify, Response
import cv2
import threading

app = Flask(__name__)

# 全局变量，用于管理录像状态
recording = False
video_writer = None
capture_device = None  # 改为动态初始化
camera_initialized = False  # 标记摄像头是否已初始化


def initialize_camera():
    """初始化摄像头"""
    global capture_device, camera_initialized
    if not camera_initialized:
        capture_device = cv2.VideoCapture(0)  # 确保索引 0 是正确的摄像头
        if not capture_device.isOpened():  # 检查摄像头是否成功打开
            raise Exception("Could not open camera")
        camera_initialized = True
def release_camera():
    """释放摄像头"""
    global capture_device, camera_initialized
    if camera_initialized and capture_device is not None:
        capture_device.release()
        camera_initialized = False

def generate_video_stream():
    """生成视频流"""
    global capture_device
    while recording and capture_device:
        success, frame = capture_device.read()
        if not success:
            break
        else:
            # 添加镜像翻转操作（水平翻转）
            mirrored_frame = cv2.flip(frame, 1)
            
            print("Frame captured:", mirrored_frame.shape)
            
            # 将翻转后的帧编码为 JPEG
            _, buffer = cv2.imencode('.jpg', mirrored_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video-stream')
def video_stream():
    """视频流路由"""
    def generate_frames():
        global capture_device
        while recording and capture_device:
            success, frame = capture_device.read()
            if not success:
                break
            else:
                # 镜像翻转视频帧
                mirrored_frame = cv2.flip(frame, 1)

                # 将视频帧调整为与蒙版相同的尺寸
                resized_frame = cv2.resize(mirrored_frame, (1920, 1090))

                # 编码为 JPEG 格式
                _, buffer = cv2.imencode('.jpg', resized_frame)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if recording and capture_device:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({'error': 'Recording has not started yet.'}), 403
def record_video(output_path):
    """录像功能"""
    global recording, video_writer, capture_device
    while recording and capture_device:
        ret, frame = capture_device.read()
        if ret:
            video_writer.write(frame)


@app.route('/start-recording', methods=['POST'])
def start_recording():
    """启动录像"""
    global recording, video_writer
    if recording:
        return jsonify({'success': False, 'error': 'Already recording'})
    
    # 初始化摄像头
    initialize_camera()

    recording = True
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = 'static/output.avi'
    video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    thread = threading.Thread(target=record_video, args=(output_path,))
    thread.start()
    return jsonify({'success': True})


@app.route('/stop-recording', methods=['POST'])
def stop_recording():
    """停止录像"""
    global recording, video_writer
    if not recording:
        return jsonify({'success': False, 'error': 'Not currently recording'})

    recording = False
    video_writer.release()
    video_writer = None

    # 释放摄像头
    release_camera()

    return jsonify({'success': True})


@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)