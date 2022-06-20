import json
import os
import os.path as osp
import warnings
import PIL.Image
import yaml
from labelme import utils
import base64


def JSON_to_IMG(json_file, save_file):
    """
    将指定目录下的json标注文件全部转换为以下文件：①jpg原图文件、 ②png标签文件
    ③其他相关说明文件 。存储在指定路径下
    :param json_file: Json格式所在文件夹
    :param save_file: 转换后的文件存储文件夹
    """
    # 如果存储路径不存在就创建文件目录
    if not osp.exists(save_file):
        os.mkdir(save_file)
    # 文件目录下的所有json文件名称
    count = os.listdir(json_file)
    # 遍历目录下的所有json文件
    for i in range(0, len(count)):
        # 获取json文件全路径名称
        path = os.path.join(json_file, count[i])
        # 如果是imagedata格式文件进行读取
        if os.path.isfile(path):
            # 打开json文件
            data = json.load(open(path))
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
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

            captions = ['{}: {}'.format(lv, ln)
                        for ln, lv in label_name_to_value.items()]
            lbl_viz = utils.draw_label(lbl, img, captions)
            # 获取json文件名并将.json改为_json用作文件夹目录
            # out_dir = osp.basename(count[i]).replace('.json', '.png')
            out_pic_name = str(i) + ".png"  # 原图
            out_label_name = str(i) + "label.png"  # 标签图
            out_label_viz_name = str(i) + "_label_viz.png"  # 带标注的图片
            out_labeltxt_name = str(i) + "_label_names.txt"  # 标签名对应值
            out_info_name = str(i) + "_info.yaml"
            # 在目标文件夹下保存原始图片
            PIL.Image.fromarray(img).save(osp.join(save_file, out_pic_name))
            # 保存标签图片
            utils.lblsave(osp.join(save_file, out_label_name), lbl)
            # 保存带标注的可视化图片
            PIL.Image.fromarray(lbl_viz).save(osp.join(save_file, out_label_viz_name))
            with open(osp.join(save_file, out_labeltxt_name), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=label_names)
            with open(osp.join(save_file, out_info_name), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('完成了对Json文件: %s的IMG格式转换!' % osp.basename(count[i]))
            print("指定目录下的所有JSON文件完成转换！")


# 程序主入口

json_file = "./data/json"  # 这里json文件所在文件夹的全路径
save_file = "./data_label_all"  # 这里填写转换后文件保存的文件夹全路径
JSON_to_IMG(json_file, save_file)  # 调用上方定义的转换函数完成批量转换


