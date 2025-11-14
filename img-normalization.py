import cv2, os
import numpy as np
from dotenv import load_dotenv




#################################################
###  NORMALIZE IMAGES WITHOUT MODIFICATIONS   ###
#################################################




print("Stating ...")
load_dotenv()


#PATHS AND CONFIG
input_Dir = os.getenv("INPUT_DIR")
output_Dir = os.getenv("OUTPUT_DIR")
os.makedirs(output_Dir, exist_ok=True)
target_size = (140,140)



### Normalization
def normalization_Process(image_Path, save_Path=None):
    #Image path load
    image = cv2.imread(image_Path)
    
    if image is None:
        print("Error to load the image")
        return
    
    #sizing the img
    resized_Img = cv2.resize(image, target_size)
    #Scaling the img
    normalaized_Img = resized_Img / 255.0
    #Save as array

    if save_Path:
        np.save(save_Path, normalaized_Img)
        print(f"SAVED IN {save_Path}")

    return normalaized_Img


### Files processing
for file_Name in os.listdir(input_Dir):
    if file_Name.endswith(".jpg") or file_Name.endswith(".png"):
        input_Path = os.path.join(input_Dir,file_Name)
        output_Path = os.path.join(output_Dir, file_Name.split('.')[0]+ ".npy")
        normalized_Img = normalization_Process(input_Path, output_Path)



print("Done!")


