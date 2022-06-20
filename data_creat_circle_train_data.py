import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml
from labelme import utils
import base64
import cv2

def JSON_to_IMG(json_file, save_file, file_handle):
    count = os.listdir(json_file)
    for i in range(0, len(count)):
        img_name = count[i][:-5]
        path = os.path.join(json_file, count[i])
        # 如果是imagedata格式文件进行读取
        if os.path.isfile(path):
            # 打开json文件
            data = json.load(open(path))
            if data['imageData']:  # 如果imgdata存放在json文件中
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])  # json文件中有img路径
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            # 将图片中背景赋值为0
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            cv2.imwrite("./2.jpg", lbl)
            label_img = cv2.imread("./2.jpg", cv2.IMREAD_GRAYSCALE)
            ret, mask = cv2.threshold(label_img, 0.5, 255,cv2.THRESH_BINARY)

            out_pic_name = str(img_name) + ".png"  # 原图
            out_label_name = str(img_name) + "label.png"  # 标签图
            # 在目标文件夹下保存原始图片
            PIL.Image.fromarray(img).save(osp.join(save_file, out_pic_name))
            # 保存标签图片
            cv2.imwrite(osp.join(save_file, out_label_name), mask)
            # utils.lblsave(osp.join(save_file, out_label_name), lbl)
            file_handle.writelines(str(save_file)+str(out_pic_name)+ " " + str(save_file)+ str(out_label_name) + "\n" )
            print('完成了对Json文件: %s的IMG格式转换!' % osp.basename(count[i]))
            print("指定目录下的所有JSON文件完成转换！")

label_txt = "./data_train/label_path.txt"
file_handle = open(label_txt, "w")

json_file = "./data/json"
save_file = "./data_train/img_and_label"
JSON_to_IMG(json_file, save_file, file_handle)
file_handle.close()

