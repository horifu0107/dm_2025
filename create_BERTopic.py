# from bertopic import BERTopic
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import os
# from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import TfidfVectorizer
# os.environ["MECABRC"] = "/opt/homebrew/opt/mecab"
# import fugashi
# from bertopic.vectorizers import ClassTfidfTransformer
# from bertopic.representation import MaximalMarginalRelevance
# from sklearn.feature_extraction.text import CountVectorizer
# from umap import UMAP
# from hdbscan import HDBSCAN

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# # ====== 1. 日本語向けエンコーダ ======
# # 例: SBERT の日本語モデル
# embedding_model = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")

# # ====== 2. データ（文章リスト）======
# input_folder = f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/Annotated-FKC-Corpus-Ver.1.0/org"
# docs_csv=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/docs.csv"
# input_gikai_file=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/gikai_clean.csv"

# docs=[]



# # ====== 議会データと不満データを結合 =====
# gikai_df = pd.read_csv(input_gikai_file, encoding="utf-8-sig")

# # Utterance 列をリスト化
# gikai_docs = gikai_df["Utterance"].dropna().tolist()

# # 不満データ + 議会データを結合
# docs = docs + gikai_docs

# # ====== 3. 前処理（ベクトル化）======

# embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
# embeddings = embedding_model.encode(
#     docs,
#     show_progress_bar=True,
#     batch_size=16,
#     convert_to_numpy=True)

# # ====== 4. 前処理（次元削減）======
# umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)

# # ====== 5. 前処理（クラスタリング）======
# # cluster_model = KMeans(n_clusters= 8, random_state=42)
# cluster_model = HDBSCAN(
#     min_cluster_size=20,
#     metric="euclidean",
#     cluster_selection_method="eom",
#     prediction_data=True
# )
# # ====== 6. 前処理（トピックを代表する単語の形式）======

# # ストップワードリストを作成
# stopwords = {"自分", "私", "僕","全て","以上","ため"}

# def tokenize_jp(docs):
#     tagger = fugashi.Tagger()
#     words = [
#         word.surface
#         for word in tagger(docs)
#         if word.feature.pos1 == "名詞" and word.feature.pos2 != "数詞" and word.surface not in stopwords
#     ]
#     return words

# vectorizer_model = CountVectorizer(tokenizer=tokenize_jp, min_df=1)

# # ====== 7. 前処理（トピックを代表する単語の重みづけ）======
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# # ====== 8. 前処理（トピック表現）======
# representation_model = MaximalMarginalRelevance(diversity=0)

# # ====== 9. BERTopic の学習 ======

# ### 準備が整ったので、トピックモデルを実行する
# topic_model = BERTopic(
#     embedding_model=embedding_model,
#     umap_model=umap_model,
#     hdbscan_model=cluster_model,
#     vectorizer_model=vectorizer_model,
#     ctfidf_model=ctfidf_model,
#     representation_model=representation_model,
#     verbose=True
# )
# topics = topic_model.fit_transform(docs, embeddings)

# # topic_model = BERTopic(embedding_model=embedding_model, language="japanese")
# # topics, probs = topic_model.fit_transform(docs)

# # ====== 4. 結果の表示 ======
# output_csv=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/output_add_gikai.csv"

# print("トピック番号:", topics)
# print("\n--- 各トピックの詳細 ---")
# print(topic_model.get_topic_info())
# df = pd.DataFrame(topic_model.get_topic_info())
# df.to_csv(output_csv, index=False)


# # ====== 5. 特定トピックの単語一覧 ======
# print("\n--- トピック 0 のキーワード ---")
# print(topic_model.get_topic(0))


#/Users/horikawafuka2/Documents/class_2025/dm/後期期末/sample_BERTopic.py
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



# ====== 議会データと不満データを結合 =====
gikai_df = pd.read_csv(input_gikai_file, encoding="utf-8-sig")
df = pd.read_csv(docs_csv, encoding="utf-8-sig")

# Utterance 列をリスト化
gikai_docs = gikai_df["Utterance"].dropna().tolist()
docs=df["text"].dropna().tolist()

# 不満データ + 議会データを結合
# docs = docs + gikai_docs

# ====== 3. 前処理（ベクトル化）======

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = embedding_model.encode(docs)

# ====== 4. 前処理（次元削減）======
umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42)

# ====== 5. 前処理（クラスタリング）======
cluster_model = KMeans(n_clusters= 8, random_state=42)

# ====== 6. 前処理（トピックを代表する単語の形式）======

# ストップワードリストを作成
stopwords = {"自分", "私", "僕","全て","以上","ため"}

def tokenize_jp(docs):
    tagger = fugashi.Tagger()
    words = [
        word.surface
        for word in tagger(docs)
        if word.feature.pos1 == "名詞" and word.feature.pos2 != "数詞" and word.surface not in stopwords
    ]
    return words

vectorizer_model = CountVectorizer(tokenizer=tokenize_jp, min_df=1)

# ====== 7. 前処理（トピックを代表する単語の重みづけ）======
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True, bm25_weighting=True)

# ====== 8. 前処理（トピック表現）======
representation_model = MaximalMarginalRelevance(diversity=0)

# ====== 9. BERTopic の学習 ======

### 準備が整ったので、トピックモデルを実行する
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
    verbose=True
)
topics = topic_model.fit_transform(docs, embeddings)

# topic_model = BERTopic(embedding_model=embedding_model, language="japanese")
# topics, probs = topic_model.fit_transform(docs)

# ====== 4. 結果の表示 ======
output_csv=f"/Users/horikawafuka2/Documents/class_2025/dm/後期期末/datas/output_add_gikai.csv"

print("トピック番号:", topics)
print("\n--- 各トピックの詳細 ---")
print(topic_model.get_topic_info())
df = pd.DataFrame(topic_model.get_topic_info())
df.to_csv(output_csv, index=False)


# ====== 5. 特定トピックの単語一覧 ======
print("\n--- トピック 0 のキーワード ---")
print(topic_model.get_topic(0))