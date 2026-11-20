"""
工具呼叫 vs 無工具：觀察幻覺（hallucination）——航班查詢版

  同一個問題「七天後台北飛沖繩的單程班機」，跑兩個版本：
    A. 無 tool-call：模型只能憑訓練資料猜，沒有即時航班與票價 → 容易編造
    B. 有 tool-call：模型呼叫 search_flights（SerpApi Google Flights），拿到真實資料再回答
  觀察重點：
    - A 版有沒有「假裝知道」班次與票價？還是老實說查不到？
    - B 版回答的時間／票價，和印出的 [tool_result] 是否一致？
    - 模型自己轉換的機場代碼（例如 TPE、OKA）和日期是否合理？

前置條件：
    1. pip install requests python-dotenv
    2. .env 內設定 serpApi=<你的 SerpApi 金鑰>（此處為佔位符，請勿把真實金鑰寫進程式碼）
"""
import json
import os
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

from common import client, TOOL_MODEL, section, HERE

load_dotenv(os.path.join(HERE, ".env"))
serpApi_KEY = os.environ.get("serpApi", "")

SERPAPI_URL = "https://serpapi.com/search.json"

TODAY = date.today()
TARGET_DATE = TODAY + timedelta(days=7)          # 「七天後」由程式算好，不讓模型做日期運算

QUESTION = "七天後台北飛沖繩的單程班機有哪些？"
SYSTEM = {"role": "system",
          "content": (f"你是繁體中文旅遊助理。今天日期是 {TODAY.isoformat()}，"
                      f"「七天後」是 {TARGET_DATE.isoformat()}。"
                      "查詢航班時，請自行把地點轉成機場 IATA 代碼。"
                      "若工具查不到資料，請老實說查不到，不要編造。回答 150 字以內。")}


# ===== 工具本體：真正執行的是本地 Python，不是模型 =====
def search_flights(departure_id: str, arrival_id: str, outbound_date: str) -> dict:
    """search one-way flights with SerpApi Google Flights"""
    resp = requests.get(
        SERPAPI_URL,
        params={
            "engine": "google_flights",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date,
            "type": 2,              # 2 = 單程（1 = 來回）
            "currency": "TWD",
            "hl": "zh-TW",
            "gl": "tw",
            "api_key": serpApi_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:                          # SerpApi 查無結果時會回傳 error 欄位
        return {"error": data["error"]}

    # 把原始結果壓成精簡結構，省 token 也比較好讀（只取前 5 筆）
    itineraries = (data.get("best_flights") or []) + (data.get("other_flights") or [])
    result = []
    for it in itineraries[:5]:
        segments = it.get("flights", [])
        if not segments:
            continue
        result.append({
            "price_twd": it.get("price"),
            "total_duration_min": it.get("total_duration"),
            "stops": max(len(segments) - 1, 0),
            "depart": segments[0]["departure_airport"]["time"],
            "arrive": segments[-1]["arrival_airport"]["time"],
            "segments": [
                {
                    "airline": s.get("airline"),
                    "flight_number": s.get("flight_number"),
                    "from": s["departure_airport"]["id"],
                    "to": s["arrival_airport"]["id"],
                }
                for s in segments
            ],
        })

    if not result:
        return {"error": "查無航班資料"}

    return {"fields": "price_twd=票價(新台幣), total_duration_min=總飛行分鐘, stops=轉機次數",
            "query": {"from": departure_id, "to": arrival_id, "date": outbound_date},
            "data": result}


TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "查詢指定日期的單程航班與票價。僅用於航班問題，天氣等問題不要使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "departure_id": {"type": "string",
                                 "description": "出發機場 IATA 代碼，例如 TPE（桃園）、TSA（松山）"},
                "arrival_id": {"type": "string",
                               "description": "抵達機場 IATA 代碼，例如 OKA（那霸）"},
                "outbound_date": {"type": "string",
                                  "description": "出發日期，格式 YYYY-MM-DD，必須是今天或之後"},
            },
            "required": ["departure_id", "arrival_id", "outbound_date"],
        },
    },
}]
TOOL_FUNCS = {"search_flights": search_flights}


# ===== A. 無 tool-call =====
def ask_without_tool(question):
    r = client.chat.completions.create(
        model=TOOL_MODEL, messages=[SYSTEM, {"role": "user", "content": question}])
    return r.choices[0].message.content


# ===== B. 有 tool-call =====
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
    return resp2.choices[0].message.content


if __name__ == "__main__":
    if not serpApi_KEY:
        raise SystemExit("請先在 .env 設定 serpApi")

    section("A. 無 tool-call（憑空回答）")
    print(ask_without_tool(QUESTION))

    section("B. 有 tool-call（查詢 SerpApi Google Flights）")
    print(ask_with_tool(QUESTION))