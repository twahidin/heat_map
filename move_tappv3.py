import streamlit as st #success this version does not support linear regression
# To make things easier later, we're also importing numpy and pandas for
# working with sample data. Need to upload this version to an ipad but we need to cut down on the face landmarks to save processing power
import cv2
import av
import os
import re
import threading
from typing import Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import pandas as pd
import copy 
#from progress.bar import Bar


#from gsheetsdb import connect
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)


# EDGES = {
#     (0, 1): 'm',
#     (0, 2): 'c',
#     (1, 3): 'm',
#     (2, 4): 'c',
#     (0, 5): 'm',
#     (0, 6): 'c',
#     (5, 7): 'm',
#     (7, 9): 'm',
#     (6, 8): 'c',
#     (8, 10): 'c',
#     (5, 6): 'y',
#     (5, 11): 'm',
#     (6, 12): 'c',
#     (11, 12): 'y',
#     (11, 13): 'm',
#     (13, 15): 'm',
#     (12, 14): 'c',
#     (14, 16): 'c'
# }



#commented out code for reference to be folded
# Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# Create a connection object.
# conn = connect()

# # Perform SQL query on the Google Sheet.
# # Uses st.cache to only rerun when the query changes or after 10 min.
# @st.cache(ttl=600)
# def run_query(query):
#     rows = conn.execute(query, headers=1)
#     return rows


# def draw_connections(frame, keypoints, edges, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
#     for edge, color in edges.items():
#         p1, p2 = edge
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
        
#         if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
#             cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


# def draw_keypoints(frame, keypoints, confidence_threshold):
#     y, x, c = frame.shape
#     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
#     for kp in shaped:
#         ky, kx, kp_conf = kp
#         if kp_conf > confidence_threshold:
#             cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)



# # Function to loop through each person detected and render
# def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
#     for person in keypoints_with_scores:
#         draw_connections(frame, person, edges, confidence_threshold)
#         draw_keypoints(frame, person, confidence_threshold)

def radar_chart(row1, row2, row3):  
    df = pd.DataFrame(dict(
    r=[row1,
       row2,
       row3],
    theta=['Preparation','Swing','Finished',
           ]))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    st.write(fig)

def row_style(row):
    if row.Name != 'Total':
        if row.Status == 'Intervention':
            return pd.Series('background-color: red', row.index)
        else:
            return pd.Series('background-color: green', row.index)
    else:
        return pd.Series('', row.index)



def atoi(text):
    # A helper function to return digits inside text
    return int(text) if text.isdigit() else text


def natural_keys(text):
    # A helper function to generate keys for sorting frames AKA natural sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def make_video(image_folder, video_name):
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    #print("Images "  + str(images))
    images.sort(key=natural_keys)
    #print("sort " + str(images))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    height, width, layers = frame.shape

    #fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    fourcc = cv2.VideoWriter_fourcc(*"H264")

    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
    #bar = Bar('Creating Video', max=len(images))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        #bar.next()

    #cv2.destroyAllWindows()
    #video.release()

    for file in os.listdir(image_folder):
        os.remove(image_folder + file)


#load model 

# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# movenet = model.signatures['serving_default']


st.title('MOVE Teacher Dashboard Prototype V3')

#access the database and send the data to google sheet


class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })


class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })

stream_df = pd.DataFrame({
  'str column': ['Express', 'Normal Academic', 'Normal Tech'],
  })


with st.sidebar:
    code = st.text_input('Class Code')
    name = st.text_input('Name')
    age = st.slider('Age', min_value = 12, max_value = 17, value = 15, step=1)
    gender = st.radio('Gender;',('Male','Female'))
    level = st.selectbox('Select your level:',class_df['sec column'])
    stream = st.selectbox('Select your stream:', stream_df['str column'])
    class_no = st.selectbox('Select your class:',class_df['third column'])



live_code = st.text_input('Please enter the current close code for live feedback:')


st.subheader('Class Analysis for class: ' + str(live_code))

# class OpenCVVideoProcessor(VideoProcessorBase):
#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         frame = frame.to_ndarray(format="bgr24")

        



#         # # Resize image
#         # img = in_image.copy()
#         # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
#         # input_img = tf.cast(img, dtype=tf.int32)
        
#         # # Detection section
#         # results = movenet(input_img)
#         # keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
#         # # Render keypoints 
#         # loop_through_people(in_image, keypoints_with_scores, EDGES, 0.1)

#         return av.VideoFrame.from_ndarray(frame, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#         key="opencv-filter",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         video_processor_factory=OpenCVVideoProcessor,
#         async_processing=True,
#     )




def main():

    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    num_of_frames = 100
    first_iteration_indicator = 1
    accum_image = []
    first_frame = []



    class OpenCVVideoProcessor(VideoProcessorBase):
            frame_lock: threading.Lock  # `transform()` is running in another thread, then a lock object is used here for thread-safety.
            in_image: Union[np.ndarray, None]

            def __init__(self) -> None:

                self.frame_lock = threading.Lock()
                self.in_image = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:

                in_image = frame.to_ndarray(format="bgr24")

                with self.frame_lock:
                    self.in_image = in_image

                return av.VideoFrame.from_ndarray(in_image, format="bgr24")

    ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )

    
    if st.button("Generate Heatmap", key=1):
        while first_iteration_indicator < num_of_frames + 1:
            if ctx.video_processor:
                with ctx.video_processor.frame_lock:
                    frame = ctx.video_processor.in_image

                    print(first_iteration_indicator)
                    if first_iteration_indicator == 1:
                        first_frame = copy.deepcopy(frame)
                        height, width = frame.shape[:2]
                        accum_image = np.zeros((height, width), np.uint8)
                        first_iteration_indicator +=1
                                            
                    elif first_iteration_indicator < num_of_frames:
                        filter = background_subtractor.apply(frame)  # remove the background
                        #cv2.imwrite('./data/frame.jpg', frame)
                        #cv2.imwrite('./data/diff-bkgnd-frame.jpg', filter)

                        threshold = 2
                        maxValue = 2
                        ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

                        # add to the accumulated image
                        accum_image = cv2.add(accum_image, th1)
                         
                        cv2.imwrite('./data/mask.jpg', accum_image)

                        color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_TURBO)

                        video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

                        name = "./data/frames/frame%d.jpg" % first_iteration_indicator

                        cv2.imwrite(name, video_frame)
                        first_iteration_indicator +=1
                    else:
                        first_iteration_indicator +=1

        color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
        result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

        # save the final heatmap
        cv2.imwrite('diff-overlay.jpg', result_overlay)
        st.subheader("Heat Map Image")
        st.image('diff-overlay.jpg')

    if st.button("Make Video", key=2):

        make_video('./data/frames/', './data/output.mp4')

    if st.button("Play Video", key=3):
        video_file = open('./data/output.mp4', 'rb')
        st.video(video_file)       

    if st.button("Live Data", key=4):

        sheet_url = st.secrets["public_gsheets_url"]
        # rows = run_query(f'SELECT * FROM "{sheet_url}"')

        # # Print results.
        # for row in rows:
        #     st.write(f"{row.name} has a :{row.pet}:")

        class_data = pd.read_excel(sheet_url)

        df = class_data.loc[:,'Class':'Status']

        student_list = ['All'] + class_data['Name'].unique().tolist()

        #st.dataframe(df,1000,700)

        st.dataframe(df.style.apply(row_style, axis=1),1000,700)

    #Sample Data 

    # df = pd.DataFrame(
    #     np.random.randn(200, 3),
    #     columns=['a', 'b', 'c'])
    # c = alt.Chart(df).mark_circle().encode(
    #     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

    # val = st.slider('Select value',0,10,1)

    if st.button("Analyse", key=5):

        #st.write(c)
        
        no_prep = (df['Preparation'] > 0.7).sum()
        no_swing = (df['Swing'] > 0.7).sum()
        no_finished = (df['Finished'] > 0.7).sum()

        #print(gpus)

        st.subheader('Post Class Analysis')
        radar_chart(no_prep, no_swing, no_finished)        

if __name__ == "__main__":
    main()
    






    
