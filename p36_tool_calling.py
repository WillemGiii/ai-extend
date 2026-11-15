"""
工具呼叫 vs 無工具：觀察幻覺（hallucination）
執行：python p36_tool_cooling.py

  同一個問題「明天台北的天氣適合穿甚麼樣的衣服」，跑兩個版本：
    A. 無 tool-call：模型只能憑訓練資料猜，沒有即時天氣 → 容易編造
    B. 有 tool-call：模型呼叫 Taiwan_weather_forcast，拿到中央氣象署真實預報再回答
  觀察重點：
    - A 版有沒有「假裝知道」明天的氣溫、降雨？還是老實說查不到？
    - B 版回答的氣溫/降雨機率，和印出的工具原始資料是否一致？
"""
import json
import os
from datetime import date

import requests
from dotenv import load_dotenv

from common import client, TOOL_MODEL, section, HERE

load_dotenv(os.path.join(HERE, ".env"))
weatherapi_KEY = os.environ.get("weatherapi_KEY", "")

CWA_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001"
QUESTION = "明天台北的天氣適合穿甚麼樣的衣服"
SYSTEM = {"role": "system",
          "content": f"你是繁體中文生活助理，今天日期是 {date.today().isoformat()}。回答 100 字以內。"}


# ===== 工具本體：真正執行的是本地 Python，不是模型 =====
def Taiwan_weather_forcast(location: str = "臺北市") -> dict:
    """get the weather forecast"""
    resp = requests.get(
        CWA_URL,
        headers={"Authorization": weatherapi_KEY},
        params={"locationName": location},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    # 把 API 回傳壓成精簡結構，省 token 也比較好讀
    result = []
    for loc in data["records"]["location"]:
        periods = {}
        for el in loc["weatherElement"]:
            for t in el["time"]:
                key = (t["startTime"], t["endTime"])
                periods.setdefault(key, {})[el["elementName"]] = t["parameter"]["parameterName"]
        result.append({
            "location": loc["locationName"],
            "forecast": [{"start": s, "end": e, **v} for (s, e), v in periods.items()],
        })
    return {"fields": "Wx=天氣現象, PoP=降雨機率%, MinT/MaxT=最低/最高溫°C, CI=舒適度",
            "data": result}


TOOLS = [{
    "type": "function",
    "function": {
        "name": "Taiwan_weather_forcast",
        "description": "get the weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string",
                             "description": "縣市名稱，例如 臺北市、新北市、高雄市（注意用「臺」字）"},
            },
            "required": [],
        },
    },
}]
TOOL_FUNCS = {"Taiwan_weather_forcast": Taiwan_weather_forcast}


# ===== A. 無 tool-call =====
def ask_without_tool(question):
    r = client.chat.completions.create(
        model=TOOL_MODEL, messages=[SYSTEM, {"role": "user", "content": question}])
    return r.choices[0].message.content


# ===== B. 有 tool-call（對應投影片程式碼）=====
def ask_with_tool(question):
    messages = [SYSTEM, {"role": "user", "content": question}]

    resp1 = client.chat.completions.create(
        model=TOOL_MODEL, messages=messages, tools=TOOLS)
    msg = resp1.choices[0].message

    if not msg.tool_calls:                       # 模型判斷不需要用工具
        return msg.content

    messages.append(msg.model_dump(exclude_none=True))
    for tool_call in msg.tool_calls:             # 本地真正執行，可能不只一個
        args = json.loads(tool_call.function.arguments or "{}")
        print(f"[tool_call] {tool_call.function.name}({args})")
        try:
            result = TOOL_FUNCS[tool_call.function.name](**args)
        except Exception as e:                   # 失敗也要回報給模型，別讓它自己編
            result = {"error": str(e)}
        print("[tool_result]", json.dumps(result, ensure_ascii=False)[:600], "...")
        messages.append({"role": "tool",
                         "tool_call_id": tool_call.id,
                         "content": json.dumps(result, ensure_ascii=False)})

    resp2 = client.chat.completions.create(
        model=TOOL_MODEL, messages=messages, tools=TOOLS)
    final_answer = resp2.choices[0].message.content
    return final_answer


if __name__ == "__main__":
    if not weatherapi_KEY:
        raise SystemExit("請先在 .env 設定 weatherapi_KEY")

    section("A. 無 tool-call（憑空回答）")
    print(ask_without_tool(QUESTION))

    section("B. 有 tool-call（查詢氣象署）")
    print(ask_with_tool(QUESTION))
