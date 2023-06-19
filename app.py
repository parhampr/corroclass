import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import gdown

url="https://drive.google.com/uc?export=download&id=1uuD392BOqN11LCNUPUH2uxxngJt7McRF"
output="corroclassVGGT.h5"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, r'trained_model/corroclasstuned')
file_destination = f'{final_directory}/corroclassVGGT.h5'
if not os.path.exists(final_directory) or not os.path.isfile(file_destination):
   os.makedirs(final_directory)
   gdown.download(url, file_destination, quiet=False)

# Load the trained model from the file
model = load_model(file_destination)

# Define the list of categories
categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D']
processingInProgressMessageImage = "Upload an image to categorize"
def predict(image):
    if image is None or not image.any():
        return processingInProgressMessageImage
    # Convert the image to a numpy array
    x = np.array(image)

    # Resize the image
    x = np.resize(x, (224, 224, 3))

    # Expand the dimensions of the array
    x = np.expand_dims(x, axis=0)

    # Normalize the pixel values
    x = x / 255.0

    # Generate a prediction for the image
    y_pred = model.predict(x)

    # Convert the prediction to a binary label
    y_pred_label = np.argmax(y_pred, axis=1)

    # Get the name of the predicted category
    predicted_category = categories[y_pred_label[0]]

    return predicted_category

max_frames_in_video = 0

logo = "data:image/png;base64,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"
titleHeader=f'''
  <header class="header-class">
    <img style="width: 180px; height: auto" alt="Logo" src={logo} />
    <div class="menu">
      <div class="menu-items">Home</div>
      <div class="menu-items">
        About
        <span class="tooltiptext"
          >Revolutionising corrosion categorization using Artificial
          intelligence models to optimize maintenance strategies and repairs
          based on accurate corrosion predictions</span
        >
      </div>
      <div class="menu-items">
        Contact <span class="tooltiptext">Vishwesh - +60 10-284 5413</span>
      </div>
    </div>
  </header>
'''
cssLines ='''
      .contain {
        margin-top: 90px;
      }

      .header-class {
        background-color: #193050;
        padding: 16px;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        align-items: center;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        box-shadow: rgb(0 0 0 / 27%) 0px 4px 3px 2px;
        z-index: 500;
      }
      .menu {
        display: flex;
        justify-content: center;
        align-items: center;
        color: #fff !important;
        font-weight: bolder;
      }

      .menu-items {
        margin-right: 25px;
        font-size: 20px;
        cursor: pointer;
        color: #fff !important;
        position: relative;
        display: inline-block;
      }

      .menu-items .tooltiptext {
        visibility: hidden;
        min-width: 180px;
        font-size: 12px;
        background-color: #212121;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px 0;
        font-weight: normal;
        position: absolute;
        z-index: 10000;
        top: 210%;
        left: 50%;
        transform: translateX(-50%);
      }

      .menu-items:hover .tooltiptext {
        visibility: visible;
      }

      table {
        width: 100%;
        border-collapse: collapse !important;
        margin-top: 30px !important;
        border: 1px solid #000 !important;
      }
      
      th,
      td {
        border: 2px solid black !important;
        padding: 10px;
        text-align: center;
      }
      th {
        background-color: #e0e7ff;
        color: #6366f1;
        font-weight: bold;
      }

      td,
      th {
        padding: 12px !important;
        text-align: center !important;
      }

      h1 {
        margin-top: 18px;
      }
      
      #image_output{
          height: 350px;
          width: auto;
          margin: auto;
      }
      
      .tab-nav > button {
          font-weight: bolder;
      }
      
      .tab-nav > button.selected, .tabitem {
          box-shadow: rgba(0, 0, 0, 0.27) 1px 0px 0px 0px;
      }
      
      
      .tabitem button {
          color: #FFF !important;
          background-color: #193050 !important;
      }
'''

adminID="admin"
adminPassword="password"
authMessage='''Please login to access Corroclass'''
inputLabelTab1 = "Upload Video Here"
outputLabelTab1 = "Process Status "
inputSliderLabelTab2 = "Select Frame "
inputTextLabelTab2 = "Displaying first frame within nth second "
outputLabelTab2 = "Output"
inputLabelTab3 = "Upload Image Here"
outputLabelTab3 = "Corrosion belongs to category "
titleBeforeInputs="# Input Corroded Image/Video"
titleBeforeLookuptable = "# Look-Up Table"
lookTable=f'''{titleBeforeLookuptable}
| Corrosion Category | Scale Thickness | Estimated Metal Loss |
|--------------------|-----------------|---------------------|
| Category A         | 1-2 mm          | <1 mm               |
| Category B         | 2-3 mm          | 1-2 mm              |
| Category C         | 3-5 mm          | 2-3 mm              |
| Category D         | > 5 mm          | > 5 mm              |'''

mainAppTitle = "CorroClass"
tab1Title="Video Classification"
tab2Title="Video Classification Output"
tab3Title="Image Classification"
saveStateImageKey = "listImages"
saveStateFrameKey = "maxFrames"
processingInProgressMessageVideo = "Upload a video to categorize different corrosions"
processingSuccess = "Video was processed successfully. Proceed to video classification output tab to see the output"
buttonTextTab1="Classify Video"
buttonTextTab3="Classify Image"
outputTextError = "Video has not been processed yet. Please wait !!"
outputTextInstruction = "Please drag slider to view image frame by frame starting from 1"
def extract_first_frame_per_second(state, video_path):
    if not video_path:
        return [state, processingInProgressMessageVideo]
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))  # get the frames per second
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total number of frames
    total_seconds = total_frames // fps  # calculate total seconds of the video
    state[saveStateFrameKey] = total_seconds
    
    listImage = []
    for sec in range(total_seconds):
        # set frame position to the first frame of each second
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        success, image = vidcap.read()
        if success:
            # Call Main Machine Learning Model
            categoryItBelongs = predict(image)
            # Convert color space from RGB to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # calculate the position for the text (text_position)
            height, width = image.shape[:2]
            text_position = (width - 200, height - 20)  # adjust coordinates for larger font  
            
            # add a black rectangle as a background for the text
            cv2.rectangle(image, (text_position[0] - 20, text_position[1] + 40), 
                          (text_position[0] + 250, text_position[1] - 40), 
                          (0, 0, 0), -1)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5  # decrease font scale for smaller image
            font_color = (255, 255, 0)  # cyan color in BGR
            thickness = 3
            
            cv2.putText(image,categoryItBelongs, text_position, font, font_scale, font_color, thickness);
        else:
            print(f"Failed to capture frame at second {sec}")
            # create an empty image (all black)
            image = np.zeros((720, 1280, 3), np.uint8)

        # save the image
        
        listImage.append(image)

    vidcap.release()
    state[saveStateImageKey] = listImage
    return [state, processingSuccess]
    
theme = gr.themes.Soft(
    spacing_size="md", 
    radius_size="none", 
    text_size="lg", 
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"])


with gr.Blocks(css=cssLines, title=mainAppTitle, theme=theme) as demo:
    gradioState = gr.State({saveStateFrameKey : max_frames_in_video, saveStateImageKey: None})
    
    def click_on_output_tab(state, k):
        k = int(k)
        if state[saveStateFrameKey] < 1:
            return [gr.Slider.update(visible=False), gr.Text.update(visible=True, value=outputTextError),gr.Image.update(visible=False)]
        else:
            return [gr.Slider.update(visible=True, minimum=0, maximum=state[saveStateFrameKey], label=f"{inputSliderLabelTab2} (Total Frames : {state[saveStateFrameKey]})"), gr.Text.update(visible=True, value=outputTextInstruction if k == 0 else f"{k} second"),gr.Image.update(visible=k != 0)]
    
    def change_frame_button(k, state):
        k = int(k)
        text_value = ""
        image_value=None
        if k == 0:
            text_value = outputTextInstruction
        else:
            text_value = f"{k} second"
            image_value = state[saveStateImageKey][k-1]
        return [gr.Text.update(visible=True, value=text_value), gr.Image.update(visible=k!=0, value=image_value)]
        
        
    gr.Markdown(titleBeforeInputs, elem_id="markdown_title")
    gr.HTML(titleHeader)
    with gr.Tab(tab3Title):
        with gr.Row():
            image_tab3_input = gr.Image(label=inputLabelTab3)
            text_tab3_output = gr.Text(label=outputLabelTab3)
        image_classification_button = gr.Button(buttonTextTab3)
        
    with gr.Tab(tab1Title):
        with gr.Row():
            video_input = gr.Video(label=inputLabelTab1)
            text_output = gr.Text(label=outputLabelTab1, value=processingInProgressMessageVideo)
        video_classification_button = gr.Button(buttonTextTab1)
        
    with gr.Tab(tab2Title) as outputTab:
        frame_slider = gr.Slider(0, 0, value=0, step=1, label=inputSliderLabelTab2, visible=False)
        text_error_output = gr.Text(label=inputTextLabelTab2, value=outputTextError)
        image_output = gr.Image(label=outputLabelTab2, elem_id="image_output", visible=False, )
        
    outputTab.select(click_on_output_tab, [gradioState, frame_slider], [frame_slider, text_error_output, image_output ])
    frame_slider.change(change_frame_button, [frame_slider, gradioState], [text_error_output, image_output])
    video_classification_button.click(extract_first_frame_per_second, inputs=[gradioState, video_input], outputs=[gradioState, text_output])
    image_classification_button.click(predict, image_tab3_input, text_tab3_output)
    gr.Markdown(lookTable)


demo.launch(
    inbrowser=True,
    share=True,
    auth=(adminID, adminPassword),auth_message=authMessage,
    server_name="0.0.0.0",
    server_port=9099
)