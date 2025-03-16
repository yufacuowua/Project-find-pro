import os
import sys
import subprocess
import requests  # 添加用于 deepseek 请求
import json     # 添加用于处理 JSON

def setup_virtual_environment():
    """设置并激活虚拟环境"""
    # 获取当前脚本所在目录的虚拟环境路径
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    
    # 如果虚拟环境不存在，创建并安装依赖
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
        
        # 构建pip命令路径
        pip_cmd = os.path.join(venv_path, 'bin', 'pip')
        
        # 安装所需的包
        print("Installing required packages...")
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([pip_cmd, 'install', 'opencv-python'], check=True)
        subprocess.run([pip_cmd, 'install', 'ultralytics'], check=True)
        subprocess.run([pip_cmd, 'install', 'openai-whisper'], check=True)
        subprocess.run([pip_cmd, 'install', 'sounddevice'], check=True)
        subprocess.run([pip_cmd, 'install', 'soundfile'], check=True)
        subprocess.run([pip_cmd, 'install', 'git+https://github.com/openai/whisper.git'], check=True)
        print("All packages installed successfully!")

    # 使用虚拟环境中的Python运行主程序
    python_executable = os.path.join(venv_path, 'bin', 'python')
    if os.path.exists(python_executable):
        # 将当前脚本作为参数传递，并设置一个环境变量来防止递归
        if not os.environ.get('VENV_RUNNING'):
            env = os.environ.copy()
            env['VENV_RUNNING'] = '1'
            subprocess.run([python_executable, __file__], env=env)
            return True
    return False

# 如果不是在虚拟环境中运行，则设置虚拟环境
if not os.environ.get('VENV_RUNNING'):
    if setup_virtual_environment():
        sys.exit(0)

# 以下是原始程序代码
import cv2
from ultralytics import YOLO
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
from multiprocessing import Process, Manager
import logging

# 设置日志级别，禁止 ultralytics 输出过多信息
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def call_deepseek(prompt, model="deepseek-r1:7b"):
    """
    调用 DeepSeek API 生成回复
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "无回复")
    except Exception as e:
        return f"请求错误: {e}"

def draw_deepseek_output(frame, text, frame_height, frame_width):
    """
    在右下角1/8区域绘制 deepseek 输出
    """
    # 计算右下角区域
    h_start = frame_height * 7 // 8
    w_start = frame_width * 6 // 8
    
    # 创建半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (w_start, h_start), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # 将文本分行
    max_chars_per_line = 30
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_chars_per_line:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    if current_line:
        lines.append(' '.join(current_line))
    
    # 绘制文本
    font_scale = 0.5
    font_thickness = 1
    line_spacing = 25
    for i, line in enumerate(lines):
        y_pos = h_start + 20 + i * line_spacing
        cv2.putText(frame, line, (w_start + 10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

def record_audio(duration=5, fs=16000):
    """
    使用麦克风录音，录音时长为 duration 秒，采样率为 fs。
    录音结束后，将音频保存为临时 WAV 文件，并返回文件路径。
    """
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # 等待录音结束
    print("Recording finished.")
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio, fs)
    return temp_file.name

def get_voice_command_from_mic(duration=5, fs=16000):
    """
    录制音频并使用 Whisper 模型进行语音识别，
    返回转译后的语音命令（转换为小写）。
    使用 base 模型以提高识别准确性。
    """
    audio_path = record_audio(duration, fs)
    print("Loading Whisper base model...")
    voice_model = whisper.load_model("base")  # 改用 base 模型
    print("Transcribing audio...")
    result = voice_model.transcribe(audio_path)
    command = result["text"].strip().lower()
    return command

def person_detection_loop(model, shared_voice):
    """
    实时物体检测：
      - 打开摄像头，利用 YOLO 模型检测视频中的所有物体；
      - 为每个检测到的物体按类别分别编号；
      - 显示物体类别、编号、置信度和距离估算；
      - 可以通过语音选择任何物体作为目标；
      - 目标物体显示方向和距离提示。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera for real-time detection")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frame_area = frame_width * frame_height
    center_x = frame_width // 2
    center_threshold = 200

    # 定义距离阈值（基于边界框面积占比）
    DISTANCE_THRESHOLDS = {
        'far': 0.05,      # 边界框面积小于5%为远
        'medium': 0.15    # 边界框面积小于15%为中等，大于则为近
    }

    last_voice_command = ""
    last_deepseek_response = ""  # 存储最近的 deepseek 回复
    
    # 获取模型支持的所有类别
    class_names = model.names
    print("Supported objects:", class_names)
    
    # 用于跟踪每种物体的计数
    object_counters = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        results = model(frame)
        detections = []
        
        # 重置物体计数器
        object_counters.clear()

        # 遍历检测结果，记录每个检测到的物体
        for detection in results[0].boxes:
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
            confidence = detection.conf[0]
            class_id = int(detection.cls[0])
            label = model.names[class_id].lower()

            # 只处理置信度大于0.5的检测结果
            if confidence <= 0.5:
                continue

            # 计算边界框面积占比
            box_area = (x_max - x_min) * (y_max - y_min)
            area_ratio = box_area / frame_area

            # 判断距离
            if area_ratio < DISTANCE_THRESHOLDS['far']:
                distance = "far"
            elif area_ratio < DISTANCE_THRESHOLDS['medium']:
                distance = "middle"
            else:
                distance = "close"

            # 为每种物体单独计数
            if label not in object_counters:
                object_counters[label] = 1
            else:
                object_counters[label] += 1

            detections.append({
                "label": label,
                "index": object_counters[label],
                "bbox": (x_min, y_min, x_max, y_max),
                "center_x": (x_min + x_max) // 2,
                "confidence": confidence,
                "distance": distance,
                "area_ratio": area_ratio
            })

        # 获取语音转译的命令及目标信息
        voice_command = shared_voice.get("command", "")
        target_type = shared_voice.get("target_type", "")
        target_index = shared_voice.get("target_index", 0)

        # 如果有新的语音命令，调用 deepseek
        if voice_command and voice_command != last_voice_command:
            print("\n" + "="*50)
            print("Voice Command:", voice_command)
            if target_type:  # 如果识别到了目标物体
                prompt = f"Describe a {target_type} in a single sentence. Start directly with '{target_type.capitalize()} is' or 'A {target_type} is'. No thinking, no introduction."
                print("\nPrompt to DeepSeek:", prompt)
                response = call_deepseek(prompt)
                # 只保留描述部分，去掉思考过程
                if "think" in response.lower():
                    last_deepseek_response = response.split("think")[-1].strip()
                else:
                    last_deepseek_response = response
                print("\nDeepSeek Response:", last_deepseek_response)
                print("="*50 + "\n")
            last_voice_command = voice_command

        # 绘制检测到的每个物体
        for d in detections:
            x_min, y_min, x_max, y_max = d["bbox"]
            confidence = d["confidence"]
            label = d["label"]
            index = d["index"]
            distance = d["distance"]
            
            # 构建标签文本
            label_text = f"{label.capitalize()}, {index}({confidence:.2f}){distance}"
            
            # 检查是否是目标物体
            is_target = (label == target_type and index == target_index)
            
            if is_target:
                # 计算方向
                obj_center_x = d["center_x"]
                if abs(obj_center_x - center_x) <= center_threshold:
                    direction = "forward"
                elif obj_center_x < center_x:
                    direction = "left"
                else:
                    direction = "right"
                label_text += f" | {direction} | TARGET"
                text_color = (0, 0, 255)  # 红色：目标
                
                # 在画面底部显示目标物体的详细信息
                target_info = f"Target: {label}, {index} | {distance} | Direction: {direction}"
                cv2.putText(frame, target_info, (50, frame_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                text_color = (0, 255, 0)  # 绿色：非目标

            # 绘制边界框
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), text_color, 2)
            
            # 在边界框上方添加标签文本
            cv2.putText(frame, label_text, (x_min, y_min - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # 显示检测到的物体统计
        stats_text = " | ".join([f"{k}: {v}" for k, v in object_counters.items()])
        cv2.putText(frame, f"Detected: {stats_text}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Voice: {voice_command}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 绘制画面中心的十字标记
        cross_length = 50
        cross_color = (0, 255, 255)
        cv2.line(frame, (center_x - cross_length, frame_height // 2),
                 (center_x + cross_length, frame_height // 2), cross_color, 2)
        cv2.line(frame, (center_x, frame_height // 2 - cross_length),
                 (center_x, frame_height // 2 + cross_length), cross_color, 2)
        cv2.circle(frame, (center_x, frame_height // 2), 5, cross_color, -1)

        # 在右下角显示 deepseek 输出
        if last_deepseek_response:
            draw_deepseek_output(frame, last_deepseek_response, frame_height, frame_width)

        cv2.imshow("Real-Time Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # 加载 YOLO 模型（确保 "yolov10n.pt" 文件存在）
    model = YOLO("yolov10n.pt")

    # 定义物体名称的英文变体，包括可能的错误识别
    name_variants = {
        'person': ['person', 'people', 'human', 'per son', 'persons', 'pur son', 'per sun', 'personal', 'personnel', 'persona', 'person on', 'person in', 'person and', 'person at', 'person near', 'person by', 'person is', 'person was', 'person here', 'person there'],
        'bicycle': ['bike', 'bicycle', 'cycle', 'bi cycle', 'bicy', 'by cycle', 'buy cycle', 'bicycles', 'biking', 'cycling', 'biker', 'bikes', 'by sick', 'by sickle', 'bicycle is', 'bicycle here'],
        'car': ['car', 'auto', 'automobile', 'vehicle', 'cars', 'core', 'card', 'cart', 'car is', 'car here', 'car there', 'car at', 'car in', 'car by', 'car near', 'car was', 'car the', 'car a'],
        'motorcycle': ['motorcycle', 'motorbike', 'motor bike', 'motor cycle', 'motor', 'bike', 'motor bikes', 'motorcycles', 'motor cycling', 'motor cycles', 'motor by', 'motor is', 'motor here'],
        'airplane': ['airplane', 'plane', 'aircraft', 'air plane', 'plain', 'air craft', 'aeroplane', 'air plain', 'planes', 'airplanes', 'air planes', 'plane is', 'plane here'],
        'bus': ['bus', 'coach', 'autobus', 'boss', 'bas', 'buzz', 'bust', 'buses', 'bus is', 'bus here', 'bus stop', 'bus station', 'bus at', 'bus in', 'bus near'],
        'train': ['train', 'railway', 'tray', 'trains', 'training', 'trained', 'train is', 'train here', 'train at', 'train in', 'train station', 'train track', 'train way'],
        'truck': ['truck', 'lorry', 'track', 'trucks', 'trunk', 'drug', 'struck', 'truck is', 'truck here', 'truck at', 'truck in', 'truck near', 'truck by', 'truck was'],
        'boat': ['boat', 'ship', 'vote', 'bought', 'bot', 'boot', 'boats', 'boating', 'bout', 'both', 'boat is', 'boat here', 'boat at', 'boat in', 'boat near'],
        'traffic light': ['traffic light', 'traffic', 'light', 'traffic lights', 'traffic signal', 'traffic signals', 'traffic lamp', 'traffic lamps', 'traffic lighting'],
        'fire hydrant': ['fire hydrant', 'hydrant', 'fire hide', 'fire hiding', 'fire high', 'fire height', 'fire hydrants', 'hydrants', 'fire hide and'],
        'stop sign': ['stop sign', 'stop', 'stops', 'stop signs', 'stopping', 'stop signing', 'stop sine', 'stop signal', 'stop signals'],
        'parking meter': ['parking meter', 'parking', 'meter', 'parking meters', 'park meter', 'park meters', 'parking metre', 'parking meet'],
        'bench': ['bench', 'seat', 'beach', 'benches', 'bent', 'bunch', 'bench is', 'bench here', 'bench at', 'bench in', 'bench near'],
        'bird': ['bird', 'birdie', 'birds', 'bud', 'bert', 'birth', 'bird is', 'bird here', 'bird at', 'bird in', 'bird near', 'bird by', 'bird was'],
        'cat': ['cat', 'kitty', 'cats', 'cut', 'kit', 'catch', 'cat is', 'cat here', 'cat at', 'cat in', 'cat near', 'cat by', 'cat was', 'kitten'],
        'dog': ['dog', 'puppy', 'dogs', 'doug', 'dock', 'dark', 'dog is', 'dog here', 'dog at', 'dog in', 'dog near', 'dog by', 'dog was', 'doggy'],
        'horse': ['horse', 'horses', 'house', 'hoarse', 'horse is', 'horse here', 'horse at', 'horse in', 'horse near', 'horse by', 'horse was'],
        'sheep': ['sheep', 'ship', 'cheap', 'sheet', 'sheeps', 'sheep is', 'sheep here', 'sheep at', 'sheep in', 'sheep near', 'sheep by'],
        'cow': ['cow', 'cattle', 'how', 'cal', 'cows', 'count', 'cow is', 'cow here', 'cow at', 'cow in', 'cow near', 'cow by', 'cow was'],
        'elephant': ['elephant', 'elephants', 'elegant', 'elephant is', 'elephant here', 'elephant at', 'elephant in', 'elephant near'],
        'bear': ['bear', 'bare', 'beer', 'bears', 'beard', 'bear is', 'bear here', 'bear at', 'bear in', 'bear near', 'bear by'],
        'zebra': ['zebra', 'zebras', 'zebra is', 'zebra here', 'zebra at', 'zebra in', 'zebra near', 'zebra by', 'zebra was'],
        'giraffe': ['giraffe', 'giraff', 'draft', 'giraffes', 'giraffe is', 'giraffe here', 'giraffe at', 'giraffe in', 'giraffe near'],
        'backpack': ['backpack', 'bag', 'back pack', 'back bag', 'backpacks', 'back packs', 'pack', 'packed', 'backing'],
        'umbrella': ['umbrella', 'umbrellas', 'umbrella is', 'umbrella here', 'umbrella at', 'umbrella in', 'umbrella near'],
        'handbag': ['handbag', 'hand bag', 'bag', 'handbags', 'hand bags', 'hand back', 'hand bank', 'hand bang'],
        'tie': ['tie', 'thai', 'ty', 'ties', 'tied', 'tying', 'tie is', 'tie here', 'tie at', 'tie in', 'tie near', 'tie by'],
        'suitcase': ['suitcase', 'suit case', 'suitcases', 'suit cases', 'suit cast', 'sweet case', 'suite case'],
        'frisbee': ['frisbee', 'frisby', 'frisbees', 'frisbee is', 'frisbee here', 'frisbee at', 'frisbee in', 'frisbee near'],
        'skis': ['skis', 'ski', 'skies', 'skiing', 'skis is', 'skis here', 'skis at', 'skis in', 'skis near', 'skis by'],
        'snowboard': ['snowboard', 'snow board', 'snowboards', 'snow boards', 'snow boarding', 'snow bordered'],
        'sports ball': ['ball', 'sports ball', 'sport ball', 'balls', 'sports balls', 'sport balls', 'ball is', 'ball here'],
        'kite': ['kite', 'kites', 'tight', 'kite is', 'kite here', 'kite at', 'kite in', 'kite near', 'kite by', 'kite was'],
        'baseball bat': ['baseball bat', 'bat', 'baseball', 'baseball bats', 'base ball bat', 'base ball bats', 'baseball back'],
        'baseball glove': ['baseball glove', 'glove', 'gloves', 'baseball gloves', 'base ball glove', 'baseball love'],
        'skateboard': ['skateboard', 'skate board', 'skateboards', 'skate boards', 'skating board', 'skate boarding'],
        'surfboard': ['surfboard', 'surf board', 'surfboards', 'surf boards', 'surfing board', 'surf boarding'],
        'tennis racket': ['tennis racket', 'racket', 'tennis', 'tennis rackets', 'tennis racquet', 'tennis bracket'],
        'bottle': ['bottle', 'bottles', 'battle', 'bottle is', 'bottle here', 'bottle at', 'bottle in', 'bottle near'],
        'wine glass': ['wine glass', 'glass', 'wine', 'wine glasses', 'wine class', 'wine grass', 'wine last'],
        'cup': ['cup', 'cups', 'cap', 'cup is', 'cup here', 'cup at', 'cup in', 'cup near', 'cup by', 'cup was', 'cop'],
        'fork': ['fork', 'forks', 'four', 'fork is', 'fork here', 'fork at', 'fork in', 'fork near', 'fork by', 'folk'],
        'knife': ['knife', 'knives', 'life', 'knife is', 'knife here', 'knife at', 'knife in', 'knife near', 'nice'],
        'spoon': ['spoon', 'spoons', 'soon', 'spoon is', 'spoon here', 'spoon at', 'spoon in', 'spoon near', 'spawn'],
        'bowl': ['bowl', 'bowls', 'ball', 'bowl is', 'bowl here', 'bowl at', 'bowl in', 'bowl near', 'bold', 'bone'],
        'banana': ['banana', 'bananas', 'banana is', 'banana here', 'banana at', 'banana in', 'banana near', 'but none'],
        'apple': ['apple', 'apples', 'apple is', 'apple here', 'apple at', 'apple in', 'apple near', 'able', 'april'],
        'sandwich': ['sandwich', 'sand witch', 'sandwiches', 'sand which', 'sand witches', 'sand rich', 'some which'],
        'orange': ['orange', 'oranges', 'orange is', 'orange here', 'orange at', 'orange in', 'orange near', 'arrange'],
        'broccoli': ['broccoli', 'broccolis', 'broccoly', 'broccoli is', 'broccoli here', 'broccoli at', 'brock lee'],
        'carrot': ['carrot', 'carrots', 'care it', 'carrot is', 'carrot here', 'carrot at', 'carrot in', 'care at'],
        'hot dog': ['hot dog', 'hotdog', 'hot dogs', 'hot dogs', 'hot dock', 'hot dark', 'hot dot', 'hot dag'],
        'pizza': ['pizza', 'pizzas', 'peter', 'pizza is', 'pizza here', 'pizza at', 'pizza in', 'pizza near', 'piece'],
        'donut': ['donut', 'doughnut', 'do not', 'donuts', 'doughnuts', 'do nuts', 'don\'t', 'don it'],
        'cake': ['cake', 'cakes', 'cake is', 'cake here', 'cake at', 'cake in', 'cake near', 'take', 'bake'],
        'chair': ['chair', 'chairs', 'share', 'chair is', 'chair here', 'chair at', 'chair in', 'chair near', 'care'],
        'couch': ['couch', 'sofa', 'coach', 'couch is', 'couch here', 'couch at', 'couch in', 'couch near', 'catch'],
        'potted plant': ['plant', 'plants', 'potted plant', 'potted plants', 'planted', 'pot plant', 'pot plants'],
        'bed': ['bed', 'beds', 'bad', 'bed is', 'bed here', 'bed at', 'bed in', 'bed near', 'bet', 'bread'],
        'dining table': ['table', 'tables', 'dining table', 'dining tables', 'dinning table', 'dining stable'],
        'toilet': ['toilet', 'toilets', 'toy let', 'toilet is', 'toilet here', 'toilet at', 'toilet in', 'toy lot'],
        'tv': ['tv', 'television', 'telly', 'tv set', 'tv is', 'tv here', 'tv at', 'tv in', 'tv near', 'tv by'],
        'laptop': ['laptop', 'computer', 'lab top', 'lap top', 'laptops', 'lap tops', 'lab tops', 'lap talk'],
        'mouse': ['mouse', 'mice', 'mouth', 'mouse is', 'mouse here', 'mouse at', 'mouse in', 'mouse near'],
        'remote': ['remote', 'remote control', 're mote', 'remotes', 'remote controls', 'remote is', 'remote here'],
        'keyboard': ['keyboard', 'key board', 'keyboards', 'key boards', 'key broad', 'keyboard is', 'keyboard here'],
        'cell phone': ['phone', 'mobile', 'cell phone', 'smartphone', 'cell', 'phones', 'cell phones', 'sell phone'],
        'microwave': ['microwave', 'micro wave', 'microwaves', 'micro waves', 'microwave is', 'microwave here'],
        'oven': ['oven', 'ovens', 'ovan', 'oven is', 'oven here', 'oven at', 'oven in', 'oven near', 'open'],
        'toaster': ['toaster', 'toasters', 'toaster is', 'toaster here', 'toaster at', 'toaster in', 'toast'],
        'sink': ['sink', 'sinks', 'sync', 'sink is', 'sink here', 'sink at', 'sink in', 'sink near', 'think'],
        'refrigerator': ['refrigerator', 'fridge', 're frigerator', 'refrigerators', 'fridges', 'fridge is'],
        'book': ['book', 'books', 'look', 'book is', 'book here', 'book at', 'book in', 'book near', 'brook'],
        'clock': ['clock', 'clocks', 'clark', 'clock is', 'clock here', 'clock at', 'clock in', 'clock near'],
        'vase': ['vase', 'vases', 'face', 'vase is', 'vase here', 'vase at', 'vase in', 'vase near', 'base'],
        'scissors': ['scissors', 'scissor', 'scissors is', 'scissors here', 'scissors at', 'scissors in'],
        'teddy bear': ['teddy', 'teddy bear', 'teddy bears', 'teddy is', 'teddy here', 'teddy at', 'teddy in'],
        'hair drier': ['hair dryer', 'dryer', 'hair dry', 'hair dryers', 'hair drier', 'hair dryers', 'hair dry'],
        'toothbrush': ['toothbrush', 'tooth brush', 'toothbrushes', 'tooth brushes', 'two brush', 'tooth rush']
    }

    # 使用 Manager 创建共享字典，存储语音转译结果和目标信息
    with Manager() as manager:
        shared_voice = manager.dict()
        shared_voice["command"] = ""
        shared_voice["target_type"] = ""  # 目标物体类型
        shared_voice["target_index"] = 0  # 目标物体编号

        # 启动实时物体检测进程
        p_detection = Process(target=person_detection_loop, args=(model, shared_voice))
        p_detection.start()

        print("Real-time video detection started.")
        print("Press Enter to record audio for 5 seconds; type 'quit' to exit.")
        while True:
            user_input = input("Press Enter to record audio (or type 'quit' to exit): ")
            if user_input.strip().lower() == "quit":
                break

            # 录音并进行语音识别
            command = get_voice_command_from_mic(duration=5)
            print("Transcription result:", command)
            command = command.lower().strip()
            shared_voice["command"] = command

            # 解析语音命令中的物体类型和编号
            words = command.split()
            target_found = False
            
            # 定义数字的变体映射
            number_variants = {
                '1': ['one', 'won', 'wan', 'wun', 'want', 'once', '1st', 'first', 'single', 'wown', 'own'],
                '2': ['two', 'to', 'too', 'tue', '2nd', 'second', 'dual', 'twice'],
                '3': ['three', 'tree', 'free', '3rd', 'third', 'triple'],
                '4': ['four', 'for', 'fore', '4th', 'fourth', 'floor'],
                '5': ['five', 'fife', 'fifth', '5th', 'fifth']
            }
            
            # 反向映射，用于快速查找
            number_map = {}
            for num, variants in number_variants.items():
                for variant in variants:
                    number_map[variant] = num
                number_map[num] = num  # 添加数字本身的映射
            
            # 尝试在原始命令中查找数字
            numbers = []
            for word in words:
                # 检查是否是纯数字
                if word.isdigit():
                    numbers.append(word)
                # 检查是否在变体映射中
                elif word in number_map:
                    numbers.append(number_map[word])
                # 检查词尾是否包含数字（例如"cat1"）
                elif any(c.isdigit() for c in word):
                    # 提取词尾的数字
                    number = ''.join(c for c in word if c.isdigit())
                    numbers.append(number)
                    # 将没有数字的部分作为单独的词添加回words
                    word_without_number = ''.join(c for c in word if not c.isdigit())
                    if word_without_number:
                        words.append(word_without_number)
                # 检查单词是否包含数字变体（例如"catwon"）
                else:
                    for num, variants in number_variants.items():
                        for variant in variants:
                            if variant in word:
                                # 分离数字变体和其他部分
                                other_part = word.replace(variant, '')
                                if other_part:  # 如果还有其他部分，添加回words
                                    words.append(other_part)
                                numbers.append(num)
                                break
            
            # 首先尝试完整匹配
            for i, word in enumerate(words):
                for obj_name, variants in name_variants.items():
                    if word in variants:
                        # 查找编号
                        if numbers:  # 如果找到任何数字
                            number = numbers[0]  # 使用第一个找到的数字
                            shared_voice["target_type"] = obj_name
                            shared_voice["target_index"] = int(number)
                            target_found = True
                            print(f"Targeting {obj_name} {number}")
                            break
                if target_found:
                    break
            
            # 如果完整匹配失败，尝试组合相邻的词
            if not target_found and len(words) > 1:
                for i in range(len(words)-1):
                    combined_word = words[i] + ' ' + words[i+1]
                    for obj_name, variants in name_variants.items():
                        if combined_word in variants:
                            if numbers:  # 如果找到任何数字
                                number = numbers[0]
                                shared_voice["target_type"] = obj_name
                                shared_voice["target_index"] = int(number)
                                target_found = True
                                print(f"Targeting {obj_name} {number}")
                                break
                    if target_found:
                        break

            if not target_found:
                shared_voice["target_type"] = ""
                shared_voice["target_index"] = 0
                print("No specific target selected")

            print("shared_voice:", dict(shared_voice))

        p_detection.terminate()
        p_detection.join()

if __name__ == '__main__':
    main() 