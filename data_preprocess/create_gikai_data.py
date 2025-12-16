import json
import pandas as pd
import re

input_json = "/Users/horikawafuka2/Documents/class_2025/dm/NTCIR14-QALab-PoliInfo-FormalRunDataset/Summarization/Pref13_tokyo.json"
output_csv = "/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/gikai_clean.csv"

# 除外ルール（正規表現）
EXCLUDE_PATTERNS = [
    r"^──────────+$",
    r"^…………………………+$",
    r"^━━━━━━━━━━+$",              # 区切り線
    r"^〔.*?〕$",                    # 〔登壇〕など
    r"^（.*?）$",                    # （拍手）（笑い）
    r"^第.*?号議案$",  
    r"^諮問第.*?号$",
    r"^[一二三四五六七八九十]+番.*?君。$",  # 六番山崎一輝君。
    r"^.*?君。$",                    # 山崎一輝君。（発言者指示）
    # ===== ここから追加 =====
    r"^(平成|令和|昭和)[一二三四五六七八九十〇零]+年"
    r"[一二三四五六七八九十〇零]+月"
    r"[一二三四五六七八九十〇零]+日$",      # 和暦日付
    r"^記$",                          # 単体の「記」
    r"^([一-龥]{1,4}[ 　]+){1,}[一-龥]{1,4}$"  # 人名の羅列
]

def is_meaningful_utterance(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    for pattern in EXCLUDE_PATTERNS:
        if re.match(pattern, text):
            return False
    return True

# JSON 読み込み
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for record in data:
    speaker = record.get("Speaker", "").strip()
    utterance = record.get("Utterance", "").strip()

    if is_meaningful_utterance(utterance):
        rows.append({
            "Speaker": speaker,
            "Utterance": utterance
        })

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("クリーンな議会発話CSVを作成しました:", output_csv)