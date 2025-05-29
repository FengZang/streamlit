from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import time
import threading
from queue import Queue
import datetime
import logging

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("显示追踪器", ('是', '否'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("追踪器", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_container_width=True  # 替换为新参数
                   )


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp流链接:")
    st.sidebar.caption(
        '示例: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('检测视频流中的目标'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("加载RTSP视频流错误: " + str(e))


def process_frame(conf, model, image, is_display_tracking=False, tracker=None):
    """
    处理单帧图像，进行目标检测/跟踪，并返回处理后的图像

    Args:
    - conf (float): 置信度阈值
    - model (YoloV8): YOLOv8模型实例
    - image (numpy array): 输入图像帧
    - is_display_tracking (bool): 是否显示跟踪
    - tracker: 跟踪器对象

    Returns:
    - numpy array: 处理后的图像
    """
    # 调整图像尺寸
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 执行目标检测或跟踪
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    # 绘制检测结果并返回
    return res[0].plot()


def camera_thread(config, conf, model, is_display_tracker, tracker, frame_queue, error_queue):
    """相机处理线程函数"""
    try:
        cap = cv2.VideoCapture(config["url"])
        if not cap.isOpened():
            error_queue.put(f"无法打开摄像头 {config['name']}: {config['url']}")
            return

        fps_counter = {"start": time.time(), "frames": 0}

        while True:
            ret, frame = cap.read()
            if not ret:
                error_queue.put(f"{config['name']} 读取帧失败")
                break

            # 处理帧
            processed_frame = process_frame(
                conf, model, frame,
                is_display_tracker, tracker
            )

            # 更新FPS计数器
            fps_counter["frames"] += 1
            elapsed = time.time() - fps_counter["start"]
            fps = fps_counter["frames"] / elapsed if elapsed > 0 else 0

            # 将处理后的帧放入队列
            frame_queue.put({
                "index": config["index"],
                "frame": processed_frame,
                "fps": fps,
                "name": config["name"]
            })

            # 添加小延迟，避免过度消耗CPU
            time.sleep(0.01)

    except Exception as e:
        error_queue.put(f"{config['name']} 处理出错: {str(e)}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()


def play_multiple_rtsp_streams(conf, model):
    """
    同时播放一个或两个RTSP流并进行实时目标检测

    Parameters:
        conf: YOLOv8模型的置信度
        model: YOLOv8模型实例
    """
    st.header("多摄像头实时检测")

    # 选择摄像头数量
    cam_count = st.selectbox("选择摄像头数量", [1, 2], index=1, key="cam_count")

    # 配置摄像头
    camera_configs = []
    for i in range(cam_count):
        st.subheader(f"摄像头{i + 1}配置")
        url = st.text_input(f"RTSP流链接{i + 1}:", key=f"rtsp_{i}")
        name = st.text_input(f"摄像头名称{i + 1}:", f"摄像头{i + 1}", key=f"name_{i}")
        camera_configs.append({"url": url, "name": name, "index": i})

    # 示例提示
    st.caption('示例: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')

    # 跟踪选项
    is_display_tracker, tracker = display_tracker_options()

    # 创建布局
    if cam_count == 1:
        cols = [st.container()]  # 单摄像头全宽度
        placeholders = [st.empty()]
    else:
        cols = st.columns(2)
        placeholders = [col.empty() for col in cols]

    if st.button('开始检测'):
        # 验证输入
        if any(not config["url"] for config in camera_configs):
            st.error("请填写所有摄像头的RTSP流链接")
            return

        # 状态指示器
        status = st.empty()
        error_display = st.empty()

        # 准备队列
        frame_queue = Queue(maxsize=5)  # 限制队列大小，避免内存溢出
        error_queue = Queue()

        # 创建并启动线程
        threads = []
        active_cams = 0
        for config in camera_configs:
            try:
                thread = threading.Thread(
                    target=camera_thread,
                    args=(config, conf, model, is_display_tracker, tracker, frame_queue, error_queue),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
                active_cams += 1
                placeholders[config["index"]].info(f"成功连接: {config['name']}")
            except Exception as e:
                error_display.error(f"启动摄像头 {config['name']} 线程时出错: {str(e)}")

        if active_cams == 0:
            error_display.error("没有可用的摄像头")
            return

        try:
            while active_cams > 0:
                # 处理错误队列
                errors = []
                while not error_queue.empty():
                    errors.append(error_queue.get())

                # 显示错误信息
                if errors:
                    error_display.error("\n".join(errors))
                else:
                    error_display.empty()

                # 处理帧队列
                frames = []
                while not frame_queue.empty():
                    frames.append(frame_queue.get())

                # 显示处理后的帧
                for frame_data in frames:
                    placeholders[frame_data["index"]].image(
                        frame_data["frame"],
                        caption=f"{frame_data['name']} | FPS: {frame_data['fps']:.1f}",
                        channels="BGR",
                        use_container_width=True
                    )

                # 更新状态
                active_cams = sum(1 for t in threads if t.is_alive())
                status.info(f"活动摄像头: {active_cams}/{cam_count}")

                # 添加小延迟，避免过度消耗CPU
                time.sleep(0.01)

        except Exception as e:
            st.error(f"检测过程中出错: {str(e)}")
        finally:
            # 停止所有线程（线程为守护线程，主程序退出时会自动终止）
            st.success("检测已停止")


# 辅助类：计算FPS
class FPS:
    def __init__(self):
        self._start = time.time()
        self._frames = 0
        self._fps = 0

    def update(self):
        self._frames += 1
        now = time.time()
        elapsed = now - self._start
        if elapsed > 1:  # 每秒更新一次
            self._fps = self._frames / elapsed
            self._start = now
            self._frames = 0

    def fps(self):
        return self._fps if self._fps > 0 else 0


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('检测目标'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("加载视频错误: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "选择一个视频文件...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('检测视频中的目标'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("加载视频错误: " + str(e))

# def get_youtube_stream_url(youtube_url):
#     ydl_opts = {
#         'format': 'best[ext=mp4]',
#         'no_warnings': True,
#         'quiet': True
#     }
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(youtube_url, download=False)
#         return info['url']
#
#
# def play_youtube_video(conf, model):
#     source_youtube = st.sidebar.text_input("YouTube视频链接")
#     is_display_tracker, tracker = display_tracker_options()
#
#     if st.sidebar.button('检测目标'):
#         if not source_youtube:
#             st.sidebar.error("请输入YouTube链接")
#             return
#
#         try:
#             st.sidebar.info("提取视频流链接...")
#             stream_url = get_youtube_stream_url(source_youtube)
#
#             st.sidebar.info("正在打开视频...")
#             vid_cap = cv2.VideoCapture(stream_url)
#
#             if not vid_cap.isOpened():
#                 st.sidebar.error(
#                     "无法打开视频，请尝试其他视频.")
#                 return
#
#             st.sidebar.success("打开成功!")
#             st_frame = st.empty()
#             while vid_cap.isOpened():
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(
#                         conf,
#                         model,
#                         st_frame,
#                         image,
#                         is_display_tracker,
#                         tracker
#                     )
#                 else:
#                     break
#
#             vid_cap.release()
#
#         except Exception as e:
#             st.sidebar.error(f"发生错误: {str(e)}")
