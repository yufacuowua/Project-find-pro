import requests
import json

def call_deepseek(prompt, model="deepseek-r1:7b"):
    """
    调用 DeepSeek API 生成回复

    参数：
        prompt: 要发送的提示文本
        model: 使用的模型名称（默认 deepseek-r1:7b，根据实际情况修改）
    返回：
        模型回复的文本，如果出错则返回错误信息
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False  # 使用非流式响应
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 检查响应状态
        data = response.json()
        return data.get("response", "无回复")
    except Exception as e:
        return f"请求错误: {e}"

if __name__ == "__main__":
    print("欢迎使用 DeepSeek 对话系统（输入 exit 或 quit 退出）")
    while True:
        prompt = input("请输入提示信息：")
        if prompt.strip().lower() in ["exit", "quit"]:
            print("退出对话系统")
            break
        reply = call_deepseek(prompt)
        print("DeepSeek 回复：", reply)
