import os
import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import sys
import cv2
import numpy as np
import torchvision
from torchvision.models import mobilenet_v2
import dlib
from contextlib import contextmanager

sys.path.append('gan-control/src') 
from gan_control.inference.controller import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"

age_checkpoint_path = "age_checkpoint/epoch044_0.02343_3.9984.pth"
# celeb ahq
def get_model(model_name="se_resnext50_32x4d", num_classes=101, pretrained="imagenet"):
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.last_linear = nn.Linear(dim_feats, num_classes)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    return model

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()

# def yield_images():
#     with video_capture(0) as cap:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#         ret, img = cap.read()
        
#         if not ret:
#             raise RuntimeError("Failed to capture image")

#         yield img, None


def yield_images(specific_image):
    # with video_capture(0) as cap:
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # ret, img = cap.read()
        
        # if not ret:
        #     raise RuntimeError("Failed to capture image")
    img = specific_image
    yield img, None

def predict(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        scores = model(image_tensor)
    _, predicted_class = scores.max(1)
    
    return predicted_class.item()

def generate(specific_image, batch_size_images):
    print ('Generate GAN by: ', device)
    model = get_model(model_name="se_resnext50_32x4d", pretrained=None)
    checkpoint = torch.load(age_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)

    model.eval()
    margin = 0.4
    detector = dlib.get_frontal_face_detector()
    img_size = 512
    image_generator = yield_images(specific_image)

    gender_checkpoint_path = 'gender-checkpoint/best_checkpoint.pth'
    gender_model = mobilenet_v2(pretrained = False) #0 for female and 1 for male
    gender_model.load_state_dict(torch.load(gender_checkpoint_path))
    gender_model = gender_model.to(device)

    test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    with torch.no_grad():
        for img, name in image_generator:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_h, img_w, _ = np.shape(input_img)

            # detect faces using dlib detector
            detected = detector(input_img, 1)
            faces = np.empty((len(detected), img_size, img_size, 3))

            if len(detected) > 0:
                for i, d in enumerate(detected):
                    x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                    xw1 = max(int(x1 - margin * w), 0)
                    yw1 = max(int(y1 - margin * h), 0)
                    xw2 = min(int(x2 + margin * w), img_w - 1)
                    yw2 = min(int(y2 + margin * h), img_h - 1)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                    faces[i] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1], (img_size, img_size))

                # predict ages
                inputs = torch.from_numpy(np.transpose(faces.astype(np.float32), (0, 3, 1, 2))).to(device)
                outputs = F.softmax(model(inputs), dim=-1).cpu().numpy()
                ages = np.arange(0, 101)
                predicted_ages = (outputs * ages).sum(axis=-1)
                predicted_age = np.round(predicted_ages)
                print ("Predicted Age: ", predicted_age.item())
                
                # draw results
                for i, d in enumerate(detected):
                    label = "{}".format(np.round(predicted_ages))
                    draw_label(img, (d.left(), d.top()), label)

            else: 
                print("No face detected")

            tensor_age = torch.from_numpy(predicted_age)
            # cv2.imshow("Predicted Age", img)
            # key = cv2.waitKey(3000)

            # if key == 27:  
            #     break
            cv2.destroyAllWindows()
            ten_img = Image.fromarray(img)
            tens_img = test_transform(ten_img)
            tensor_img = torch.unsqueeze(tens_img, 0)
            gender = predict(gender_model, tensor_img, device)
    if gender == 1:
        print("Gender of source: Male")
    else:
        print("Gender of source: Female")
    controller_path = 'gan-control/resources/gan_models/controller_age015id025exp02hai04ori02gam15_temp'
    controller = Controller(controller_path)

    batch_size = 1
    truncation = 0.7
    resize = 224
    age_control = tensor_age
    pose_control = torch.tensor([[0., 0., 0.]])
    attributes_df = pd.read_pickle('gan-control/resources/ffhq_1K_attributes_samples_df.pkl')
    expressions = attributes_df.expression3d.to_list()
    expression_0 = torch.tensor([expressions[2]])
    output_img_ls = []
    for i in range(batch_size_images):
            initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(batch_size=batch_size, truncation=truncation)

            image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, age = age_control, orientation=pose_control, expression = expression_0)
            img = controller.make_resized_grid_image(image_tensors, resize=resize, nrow=1)
            img_t = test_transform(img)
            img_tensor = torch.unsqueeze(img_t, 0)

            prediction = predict(gender_model,img_tensor, device)
            print ('Female: 0, Male: 1')
            print("Gender of source: ", gender)
            print("Initial Prediction: ", prediction)

            while prediction != gender:
                initial_image_tensors, initial_latent_z, initial_latent_w = controller.gen_batch(batch_size=batch_size, truncation=truncation)
                image_tensors, _, modified_latent_w = controller.gen_batch_by_controls(latent=initial_latent_w, input_is_latent=True, age = age_control, orientation=pose_control, expression = expression_0)
                img = controller.make_resized_grid_image(image_tensors, resize=resize, nrow=1)
                img_t = test_transform(img)
                img_tensor = torch.unsqueeze(img_t, 0)
                new_prediction = predict(gender_model, img_tensor, device)
                prediction = new_prediction
            output_img_ls.append(img)
    print("Final Prediction: ", prediction)
    # img.show()
    del model
    print ('Delete GAN CONTROL MODEL FROM CUDA MEMORY')
    return output_img_ls 

def get_next_image_filename(directory, extension=".png"):
    """
    Generates the next available image file name as an integer in the specified directory.
    
    Parameters:
        directory (str): The directory where the image files are stored.
        extension (str): The file extension for the image files. Default is ".png".
        
    Returns:
        str: The next available image file name as an integer.
    """
    existing_files = [f for f in os.listdir(directory) if f.endswith(extension)]
    existing_numbers = []

    for file in existing_files:
        try:
            number = int(file.replace(extension, ""))
            existing_numbers.append(number)
        except ValueError:
            pass

    if not existing_numbers:
        return f"0{extension}"

    next_number = max(existing_numbers) + 1
    return f"{next_number}{extension}"


def rename(folder_path):
    for index, filename in enumerate(os.listdir(folder_path)):
        format = filename.split('.')[-1]
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, str(index) + '.' + format))
        
def main():
    import shutil
    save_gan_folder = 'swap_source/output_real_life_GAN'
    os.makedirs (save_gan_folder, exist_ok=True)
    source_folder = 'swap_source/real_life_assessment'
    for filename in os.listdir(source_folder):
        pic_specific_path = 'swap_source/real_life_assessment/' + filename
        save_gan_image_folder = 'swap_source/output_real_life_GAN' +'/'+ os.path.basename(pic_specific_path)
        os.makedirs (save_gan_image_folder, exist_ok=True)
        
        img_format = pic_specific_path.split('.')[-1]
        shutil.copy2 (pic_specific_path, save_gan_image_folder + '/' + 'reference.' + img_format)
        
        batch_size_images = 20
        specific_image = cv2.imread(pic_specific_path)
        gan_img_pillow_ls = generate(specific_image, batch_size_images)
        
        for gan_img_pillow in gan_img_pillow_ls:
            
            im_pillow = np.array(gan_img_pillow) # turn to array 
            gan_img = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR) 
            # save_path = save_gan_image_folder + '/' + str(index) + '.png'
            save_path = save_gan_image_folder + '/' + get_next_image_filename(save_gan_image_folder)
            print (save_path)
            cv2.imwrite(save_path, gan_img)
            print ('DONE GENERATE GAN FACE')
    
if __name__ == '__main__':
    main()