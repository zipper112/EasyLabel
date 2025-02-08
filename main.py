from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import yaml
import cv2
import numpy as np
import base64

app = Flask(__name__)

# 全局变量，用于存储图片列表、当前图片索引以及采集到的参数数据
image_list = []
current_index = 0
collected_data = {
    "region_list": [],
    "zoom_bool": [],
    "color_list": [],
    "line_list": [],
    "scale_list": [],
    "place_list": [],
    "root_path": ""
}

# ----------------------- 脚本中嵌入的处理函数（内存处理版本） -----------------------

def processLabel(image, label, color, line_width):
    """
    在给定 image 上绘制矩形框，并返回绘制后的 image 以及该矩形区域的副本
    :param image: numpy.ndarray，原始图像
    :param label: [x_min, y_min, x_max, y_max]，真实图片中的坐标
    :param color: 矩形颜色，支持 'red'、'green'、'blue'
    :param line_width: 线条宽度
    :return: (image, region)
    """
    x_min, y_min, x_max, y_max = label
    # 复制矩形区域（注意 numpy 数组中，第一个维度为 y，第二个为 x）
    region = image[y_min:y_max+1, x_min:x_max+1].copy()
    # 根据颜色字符串选择 BGR 颜色
    if isinstance(color, str):
        if color == 'red':
            b, g, r = 0, 0, 255
        elif color == 'green':
            b, g, r = 0, 255, 0
        elif color == 'blue':
            b, g, r = 255, 0, 0
        else:
            b, g, r = 0, 0, 0
    else:
        b, g, r = color
    # 在原图上绘制矩形（注意：cv2.rectangle 的参数顺序为起点和终点）
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (b, g, r), line_width)
    return image, region

def processZoom(image, region, plot_place, scale, color, line_width):
    """
    将 region 放大后放置到 image 指定角落，
    同时在放大后的区域上绘制与原矩形相同颜色和线宽的矩形框。
    :param image: 当前处理图像
    :param region: 要放大的区域（未绘制矩形框）
    :param plot_place: 放置位置，取值 'top left'、'top right'、'bottom left'、'bottom right'
    :param scale: 放大倍数
    :param color: 颜色参数，同 processLabel 中的取值
    :param line_width: 线宽
    :return: 修改后的 image
    """
    # 放大 region
    region_zoom = cv2.resize(region, (0, 0), fx=scale, fy=scale)
    h, w = region_zoom.shape[:2]
    # 根据颜色选择 BGR
    if isinstance(color, str):
        if color == 'red':
            b, g, r = 0, 0, 255
        elif color == 'green':
            b, g, r = 0, 255, 0
        elif color == 'blue':
            b, g, r = 255, 0, 0
        else:
            b, g, r = 0, 0, 0
    else:
        b, g, r = color
    # 在放大后的区域上绘制矩形框（区域坐标从(0,0)到(w-1,h-1)）
    cv2.rectangle(region_zoom, (0, 0), (w - 1, h - 1), (b, g, r), line_width)
    # 将放大后的区域替换到原图的指定位置
    o_h, o_w = image.shape[:2]
    if plot_place == 'top left':
        image[0:h, 0:w] = region_zoom
    elif plot_place == 'top right':
        image[0:h, o_w - w:] = region_zoom
    elif plot_place == 'bottom left':
        image[o_h - h:, 0:w] = region_zoom
    elif plot_place == 'bottom right':
        image[o_h - h:, o_w - w:] = region_zoom
    return image

def processMultiRegionInMemory(image, region_list, color_list, line_width_list, place_list, scale_list, zoom_bool):
    """
    对 image 应用多个区域处理：
      1. 绘制矩形框；
      2. 如果 zoom_bool 为 True，则放大该区域（同时在放大区域上绘制矩形边框）并替换到 image 指定位置。
    所有操作均在内存中完成，处理后返回最终图像。
    """
    processed_img = image.copy()
    for i in range(len(region_list)):
        # 绘制矩形框，返回更新后的 image 和提取的区域（region不含矩形框）
        processed_img, region = processLabel(processed_img, region_list[i], color_list[i], line_width_list[i])
        if zoom_bool[i]:
            processed_img = processZoom(processed_img, region, place_list[i], scale_list[i], color_list[i], line_width_list[i])
    return processed_img

# ----------------------- 路由定义 -----------------------

@app.route('/')
def index():
    return render_template('index.html')

# 根据用户输入的图片文件夹加载图片
@app.route('/load_images', methods=['POST'])
def load_images():
    folder_path = request.form.get('folder_path')
    global image_list, current_index, collected_data
    if not os.path.isdir(folder_path):
        return jsonify({"error": "无效的文件夹路径"})
    allowed_ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_list = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_ext)]
    image_list.sort()
    current_index = 0
    collected_data["root_path"] = folder_path
    if not image_list:
        return jsonify({"error": "该文件夹中没有找到图片"})
    return jsonify({"filename": image_list[current_index]})

# 图片浏览接口：根据方向更新当前图片索引
@app.route('/get_image', methods=['GET'])
def get_image():
    global image_list, current_index
    direction = request.args.get('direction')
    if direction == "next" and current_index < len(image_list) - 1:
        current_index += 1
    elif direction == "prev" and current_index > 0:
        current_index -= 1
    return jsonify({"filename": image_list[current_index]})

# 返回图片内容，供前端直接访问
@app.route('/images/<filename>')
def serve_image(filename):
    folder_path = collected_data["root_path"]
    return send_from_directory(folder_path, filename)

# 保存 YAML 配置到文件（保留该接口）
@app.route('/save_yaml', methods=['POST'])
def save_yaml():
    global collected_data
    data = request.get_json()
    output_path = data.get("output_path")
    output_file = data.get("output_file")
    collected_data["region_list"] = data.get("region_list", [])
    collected_data["zoom_bool"] = data.get("zoom_bool", [])
    collected_data["color_list"] = data.get("color_list", [])
    collected_data["line_list"] = data.get("line_list", [])
    collected_data["scale_list"] = data.get("scale_list", [])
    collected_data["place_list"] = data.get("place_list", [])
    full_path = os.path.join(output_path, output_file)
    try:
        with open(full_path, 'w', encoding='utf-8') as f:
            yaml.dump(collected_data, f, allow_unicode=True)
        return jsonify({"status": "success", "message": "YAML 文件保存成功！"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# 处理当前图片，并返回处理后的图像（不保存到磁盘）
@app.route('/process_current', methods=['POST'])
def process_current():
    data = request.get_json()
    region_list = data.get("region_list", [])
    zoom_bool = data.get("zoom_bool", [])
    color_list = data.get("color_list", [])
    line_list = data.get("line_list", [])
    scale_list = data.get("scale_list", [])
    place_list = data.get("place_list", [])
    root_path = data.get("root_path", collected_data["root_path"])
    global image_list, current_index
    if not image_list:
        return jsonify({"status": "error", "message": "没有加载图片"})
    current_filename = image_list[current_index]
    img_path = os.path.join(root_path, current_filename)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"status": "error", "message": "无法读取图片"})
    # 在内存中处理图像
    processed_img = processMultiRegionInMemory(img, region_list, color_list, line_list, place_list, scale_list, zoom_bool)
    # 编码为 JPEG 并转换为 base64
    retval, buffer = cv2.imencode('.jpg', processed_img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"status": "success", "processed_image": jpg_as_text})

# 运行脚本处理整个文件夹图片（保存到物理存储），保留原有接口
@app.route('/run_script', methods=['POST'])
def run_script():
    global collected_data
    data = request.get_json()
    collected_data["region_list"] = data.get("region_list", [])
    collected_data["zoom_bool"] = data.get("zoom_bool", [])
    collected_data["color_list"] = data.get("color_list", [])
    collected_data["line_list"] = data.get("line_list", [])
    collected_data["scale_list"] = data.get("scale_list", [])
    collected_data["place_list"] = data.get("place_list", [])
    collected_data["root_path"] = data.get("root_path", collected_data["root_path"])

    
    if not os.path.isdir(collected_data["root_path"]):
        return jsonify({"status": "error", "message": "无效的图片文件夹路径"})
    
    allowed_ext = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    processed_files = []
    for image_name in os.listdir(collected_data["root_path"]):
        if image_name.lower().endswith(allowed_ext):
            input_img_path = os.path.join(collected_data["root_path"], image_name)
            output_img_path = os.path.join(data.get("root_path", collected_data["root_path"]), 'boxed_' + image_name)
            try:
                img = cv2.imread(input_img_path)
                processed_img = processMultiRegionInMemory(img,
                                                           collected_data["region_list"],
                                                           collected_data["color_list"],
                                                           collected_data["line_list"],
                                                           collected_data["place_list"],
                                                           collected_data["scale_list"],
                                                           collected_data["zoom_bool"])
                cv2.imwrite(output_img_path, processed_img)
                processed_files.append(output_img_path)
            except Exception as e:
                return jsonify({"status": "error", "message": f"处理图片 {image_name} 时出错：{str(e)}"})
    return jsonify({"status": "success", "message": "脚本运行成功，共处理图片数：" + str(len(processed_files))})

if __name__ == '__main__':
    app.run(debug=True)
