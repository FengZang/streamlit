# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper_2

# Setting page layout
st.set_page_config(
    page_title="企叮咚行为识别检测",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("企叮咚行为识别检测")

# Sidebar
st.sidebar.header("模型设置")

# Model Options
model_type = st.sidebar.radio(
    "选择任务", ['检测', '分割'])

confidence = float(st.sidebar.slider(
    "选择置信度", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == '检测':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == '分割':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper_2.load_model(model_path)
except Exception as ex:
    st.error(f"无法加载模型. 请检查路径: {model_path}")
    st.error(ex)

st.sidebar.header("图像/视频流设置")
source_radio = st.sidebar.radio(
    "选择数据源", settings.SOURCES_LIST)

source_img = None

if source_radio == settings.RTSP:
    helper_2.play_multiple_rtsp_streams(confidence, model)

elif source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "选择一张图像...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="默认图像",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="上传图像",
                         use_container_width=True)
        except Exception as ex:
            st.error("无法打开图像.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='检测结果',
                     use_container_width=True)
        else:
            if st.sidebar.button('开始检测'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='检测结果',
                         use_container_width=True)
                try:
                    with st.expander("检测结果"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("未上传图像!")

elif source_radio == settings.VIDEO:
    helper_2.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper_2.play_webcam(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper_2.play_youtube_video(confidence, model)

else:
    st.error("请选择一个有效的源类型!")
