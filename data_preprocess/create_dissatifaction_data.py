from bertopic import BERTopic
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import csv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import fugashi
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
import re
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import json
# ====== 1. 日本語向けエンコーダ ======
# 例: SBERT の日本語モデル
embedding_model = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")

# ====== 2. データ（文章リスト）======
input_folder = f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/Annotated-FKC-Corpus-Ver.1.0/org"
docs_csv=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/docs.csv"
input_gikai_file=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/gikai_clean.csv"

docs=[]

# org フォルダ内の全ファイルを処理
for fname in os.listdir(input_folder):
    if fname.startswith("."):
        continue  # .DS_Store 対策

    fpath = os.path.join(input_folder, fname)
    print(f"処理中: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        datalist = f.readlines()

        # 偶数行(1,3,5...) が本文 → strip して1行化
        for i in range(1, len(datalist), 2):
            text = datalist[i].strip().replace("\n", " ")
            docs.append(text)

# DataFrame 化して保存
docs_df = pd.DataFrame({"text": docs})
docs_df.to_csv(docs_csv, index=False)

print("保存完了:", docs_csv)

