"""
ReAct Demo：讓「觀察（Observation）」決定下一步「行動（Action）」
執行：
    python p38_Demo_ReAct.py        # 主題 + 對照題都跑
    python p38_Demo_ReAct.py 1      # 只跑主題（兩個工具，第二個依賴第一個的結果）
    python p38_Demo_ReAct.py 2      # 只跑對照題（只需要一個工具）

  主題：「我這個週末想在台北出去玩，請依照天氣預報幫我推薦適合的活動和地點。」
    第一步：查天氣（taiwan_weather_forecast）→ 觀察到「有雨」或「晴朗」
    第二步：依觀察決定搜尋詞（web_search）→「台北 雨天 室內景點」或「台北 晴天 戶外 行程」
    第三步：整合兩邊資訊回答
  → 同一個問題，天氣不同就會走出不同的搜尋詞：這就是「觀察 → 決策」，不是照劇本演出。

  對照題：「台北今天會下雨嗎？」
  → 模型查完天氣就該停，自己判斷不需要第二個工具。

  ReAct 步驟 ↔ astream_events 事件對照：
    ┌──────────────────┬──────────────────────────────┬──────────┐
    │ ReAct 步驟        │ 對應事件                       │ 終端機標籤 │
    ├──────────────────┼──────────────────────────────┼──────────┤
    │ 推理 Thought      │ on_chat_model_stream（文字）    │ 黃色      │
    │ 呼叫工具 Action    │ on_tool_start                 │ 青色      │
    │ 觀察 Observation  │ on_tool_end                   │ 紫色      │
    │ 最終回答          │ 最後一輪（沒有 tool_calls）的串流 │ 綠色      │
    └──────────────────┴──────────────────────────────┴──────────┘

  課堂練習：
    1. 改 QUESTIONS 的題目（例如「明天適合去陽明山嗎？」），觀察工具呼叫順序有沒有改變
    2. 把工具 docstring 改得很模糊（例如只寫 "search"），看模型會不會選錯工具 → 工具描述就是工具的 prompt
    3. 拿掉 system prompt 裡「呼叫工具前先說明原因」那一條，看 Thought 是不是消失了

前置條件：
    1. pip install langchain langchain-openai requests python-dotenv
    2. .env 內設定 weatherapi_KEY=<中央氣象署授權碼>、serpApi=<SerpApi 金鑰>
"""
import asyncio
import json
import os
import sys
from datetime import date

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from common import API_BASE, API_KEY, TOOL_MODEL, section, HERE

load_dotenv(os.path.join(HERE, ".env"))
weatherapi_KEY = os.environ.get("weatherapi_KEY", "")
serpApi_KEY = os.environ.get("serpApi", "")

CWA_URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/F-C0032-001"
SERPAPI_URL = "https://serpapi.com/search.json"

QUESTIONS = {
    "1": ("主題：兩個工具，第二步取決於第一步的觀察",
          "我明天想在台北出去玩，請依照天氣預報幫我推薦適合的活動和地點。"),
    "2": ("對照題：只需要一個工具",
          "台北今天會下雨嗎？"),
}

TODAY = date.today()
WEEKDAY = "一二三四五六日"[TODAY.weekday()]

# ===== System Prompt：角色、ReAct 流程、何時停止、Thought 要說出來 =====
SYSTEM_PROMPT = f"""你是繁體中文的台北旅遊助理。今天是 {TODAY.isoformat()}（星期{WEEKDAY}）。

你用 ReAct（推理 → 行動 → 觀察）的方式工作，流程如下：
1. 推理：先想清楚回答這個問題還缺什麼資訊。
2. 行動：如果缺資訊，就呼叫最適合的工具。
3. 觀察：拿到工具結果後，先用一兩句話簡述你觀察到什麼重點。
4. 再推理：根據觀察決定下一步——還需要其他工具嗎？要用什麼參數或關鍵字？
5. 回答：資訊足夠時就停止呼叫工具，直接給出最終回答。不需要的工具不要呼叫。

規則：
- 每次呼叫工具之前，一定要先輸出一兩句話，說明你為什麼要呼叫這個工具、打算查什麼。不可以不說話就直接呼叫工具。
- 每次拿到工具結果後，先簡述觀察，再說明下一步的決定。
- 不得憑空猜測天氣，天氣資訊必須透過天氣工具查詢。
- 天氣預報只涵蓋未來 36 小時；若使用者問的日期超出範圍，要老實說明，並以最接近的預報當參考。
- 搜尋關鍵字要根據天氣觀察決定（例如下雨就找室內景點，晴天就找戶外行程）。
- 若工具回傳錯誤或查不到資料，請老實說查不到，不要編造。
- 最終回答 200 字以內。"""


# ===== 工具：docstring 會成為模型看到的工具說明 → 工具描述就是工具的 prompt =====
@tool(parse_docstring=True)          # 解析 Args：參數說明也會給模型看
def taiwan_weather_forecast(location: str = "臺北市") -> str:
    """查詢中央氣象署的台灣縣市「未來 36 小時」天氣預報，包含天氣現象、降雨機率、最低／最高溫、舒適度。
    使用時機：問題牽涉天氣、下不下雨、溫度、穿著、適不適合出門時，一定要先用這個工具。
    不要用於：景點、活動、新聞等非天氣的資訊。

    Args:
        location: 縣市名稱，例如 臺北市、新北市、高雄市（注意用「臺」字）
    """
    try:
        resp = requests.get(CWA_URL, headers={"Authorization": weatherapi_KEY},
                            params={"locationName": location}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        result = []
        for loc in data["records"]["location"]:
            periods = {}
            for el in loc["weatherElement"]:
                for t in el["time"]:
                    key = (t["startTime"], t["endTime"])
                    periods.setdefault(key, {})[el["elementName"]] = t["parameter"]["parameterName"]
            result.append({"location": loc["locationName"],
                           "forecast": [{"start": s, "end": e, **v} for (s, e), v in periods.items()]})
        if not result:
            return json.dumps({"error": f"查無 {location} 的預報，請確認縣市名稱"}, ensure_ascii=False)
        return json.dumps({"fields": "Wx=天氣現象, PoP=降雨機率%, MinT/MaxT=最低/最高溫°C, CI=舒適度",
                           "data": result}, ensure_ascii=False)
    except Exception as e:                       # 失敗也回報給模型，別讓它自己編
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool(parse_docstring=True)          # 解析 Args：參數說明也會給模型看
def web_search(query: str) -> str:
    """用 Google 搜尋網路上的即時資訊，例如景點、活動、展覽、餐廳、新聞。
    使用時機：需要推薦地點或活動時。關鍵字可以搭配天氣結果決定，
    例如下雨搜尋「台北 雨天 室內景點」，晴天搜尋「台北 晴天 戶外 行程」。
    不要用於：查詢天氣（天氣請用 taiwan_weather_forecast）。

    Args:
        query: 搜尋關鍵字，建議用繁體中文
    """
    try:
        resp = requests.get(SERPAPI_URL, params={
            "engine": "google", "q": query, "hl": "zh-tw", "gl": "tw",
            "num": 5, "api_key": serpApi_KEY}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return json.dumps({"error": data["error"]}, ensure_ascii=False)
        results = [{"title": r.get("title"), "snippet": r.get("snippet")}
                   for r in (data.get("organic_results") or [])[:5]]
        if not results:
            return json.dumps({"error": "查無搜尋結果"}, ensure_ascii=False)
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


TOOLS = [taiwan_weather_forecast, web_search]


# ===== 終端機顏色：一眼看出 ReAct 迴圈 =====
if os.name == "nt":
    os.system("")                                # 讓 Windows 終端機啟用 ANSI 顏色
YELLOW, CYAN, MAGENTA, GREEN, GRAY, RESET = (
    "\033[33m", "\033[36m", "\033[35m", "\033[32m", "\033[90m", "\033[0m")


def build_agent():
    llm = ChatOpenAI(model=TOOL_MODEL, base_url=API_BASE, api_key=API_KEY, temperature=0)
    return create_agent(model=llm, tools=TOOLS, system_prompt=SYSTEM_PROMPT)


async def run_react(agent, question: str):
    print(f"{GRAY}[User]{RESET} {question}")
    round_no = 0

    async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": 12},      # 保險：避免模型無限呼叫工具
            version="v2"):
        kind = ev["event"]

        if kind == "on_chat_model_start":        # 新的一輪推理開始
            round_no += 1
            print(f"\n{YELLOW}[Thought｜第 {round_no} 輪模型輸出]{RESET} ", end="", flush=True)

        elif kind == "on_chat_model_stream":     # 推理：模型逐字串流的文字
            content = ev["data"]["chunk"].content
            if isinstance(content, str) and content:
                print(f"{YELLOW}{content}{RESET}", end="", flush=True)

        elif kind == "on_chat_model_end":        # 這一輪結束：有 tool_calls → 繼續迴圈；沒有 → 最終回答
            output = ev["data"]["output"]
            if getattr(output, "tool_calls", None):
                if not (output.content or "").strip():
                    print(f"{GRAY}（模型沒有說明理由就直接呼叫工具）{RESET}", end="")
                print()
            else:
                print(f"\n{GREEN}[Final Answer｜這一輪沒有呼叫工具 → 迴圈結束]{RESET}")

        elif kind == "on_tool_start":            # 行動：呼叫工具（名稱 + 輸入）
            args = json.dumps(ev["data"].get("input"), ensure_ascii=False)
            print(f"{CYAN}[Action] {ev['name']}({args}){RESET}")

        elif kind == "on_tool_end":              # 觀察：工具回傳
            output = ev["data"].get("output")
            text = getattr(output, "content", output)
            print(f"{MAGENTA}[Observation] {str(text)[:400]} ...{RESET}")


async def main(keys):
    agent = build_agent()
    for k in keys:
        title, question = QUESTIONS[k]
        section(title)
        await run_react(agent, question)


if __name__ == "__main__":
    if not weatherapi_KEY:
        raise SystemExit("請先在 .env 設定 weatherapi_KEY")
    if not serpApi_KEY:
        raise SystemExit("請先在 .env 設定 serpApi")

    picked = [a for a in sys.argv[1:] if a in QUESTIONS] or list(QUESTIONS)
    asyncio.run(main(picked))
