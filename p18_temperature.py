"""
temperature 實驗：同一個問題問 10 次，看答案有多「發散」
執行：python p18_temperature.py

  - temperature 越低 → 越保守，每次答案幾乎一樣
  - temperature 越高 → 越有創意，答案種類越多（太高可能開始胡言亂語）
  - top_p 固定為 1，只觀察 temperature 的影響
"""
from collections import Counter

from common import MODEL_NAME, client, section

# 開放式問題才看得出差異；限定「只回答店名」方便統計
prompt = "幫一家新開的咖啡店取名字，只回答一個店名，不要任何解釋。"

TIMES = 10                          # 每個溫度問幾次

for t in [0.0, 0.5, 1.0, 1.5]:
    section(f"temperature = {t}")

    names = []
    for _ in range(TIMES):
        for i in range(10):
            r = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=t,          # 本次實驗的主角
                top_p=1,     # 不另外截斷候選字，讓 temperature 單獨作用
                seed=i,
                max_tokens=50,          # 店名很短，限制長度省 token
                messages=[{"role": "user", "content": prompt}],
            )
            names.append(r.choices[0].message.content.strip())

    # Counter 統計每個答案出現幾次，例如 {'晨光咖啡': 7, '慢時光': 3}
    c = Counter(names)
    print(f"共 {len(c)} 種答案")
    for name, n in c.most_common():
        print(f"  {n:>2} 次  {name}")

print("\n👉 要穩定（客服、資料擷取）用低溫；要創意（取名、文案）用高溫")
