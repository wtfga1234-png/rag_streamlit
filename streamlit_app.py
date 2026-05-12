"""
Streamlit + Groq API - 8種 RAG 策略 PDF 問答系統
安裝依賴：pip install streamlit groq pypdf sentence-transformers numpy faiss-cpu scikit-learn
執行方式：streamlit run rag_streamlit.py
"""

import streamlit as st
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
import os

# ─────────────────────────────────────────────
# 頁面設定
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="多策略 RAG PDF 問答系統",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 自訂樣式
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* 全域字體 & 背景 */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main { background-color: #f8f9fb; }

    /* 標題卡片 */
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; letter-spacing: -0.5px; }
    .hero p  { margin: 0.5rem 0 0; opacity: 0.75; font-size: 1rem; }

    /* 區塊卡片 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
    }

    /* 答案區 */
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        white-space: pre-wrap;
        line-height: 1.75;
        font-size: 0.95rem;
    }

    /* 策略徽章 */
    .badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* 來源文本 */
    .source-chunk {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.85rem;
        line-height: 1.65;
        color: #374151;
    }
    .chunk-label {
        font-weight: 700;
        color: #6b7280;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    /* 狀態欄 */
    .status-ok   { color: #16a34a; font-weight: 600; }
    .status-err  { color: #dc2626; font-weight: 600; }
    .status-warn { color: #d97706; font-weight: 600; }

    div[data-testid="stExpander"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RAG 核心類別
# ─────────────────────────────────────────────
class MultiStrategyRAG:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        self.chunks: list[str] = []
        self.embeddings = None
        self.index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    # ── 載入 PDF ────────────────────────────
    def load_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            full_text = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )

            self.chunks = self._split_text(full_text, chunk_size=800, overlap=150)

            self.embeddings = self.embedding_model.encode(
                self.chunks, convert_to_numpy=True, show_progress_bar=False
            )

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings.astype("float32"))

            self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)

            return (
                f"✅ 成功載入！共 **{len(reader.pages)}** 頁，"
                f"分割為 **{len(self.chunks)}** 個片段。"
            )
        except Exception as e:
            return f"❌ 載入失敗：{e}"

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = re.sub(r'\s+', ' ', text[start:start + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ── 8 種策略 ────────────────────────────
    def strategy_1_basic_similarity(self, query: str, top_k: int = 3):
        """策略1: 基礎語意相似度搜尋"""
        qv = self.embedding_model.encode([query]).astype("float32")
        _, idxs = self.index.search(qv, top_k)
        return [self.chunks[i] for i in idxs[0]]

    def strategy_2_tfidf(self, query: str, top_k: int = 3):
        """策略2: TF-IDF 關鍵詞搜尋"""
        qv = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * qv.T).toarray().flatten()
        return [self.chunks[i] for i in scores.argsort()[-top_k:][::-1]]

    def strategy_3_hybrid(self, query: str, top_k: int = 3):
        """策略3: 混合搜尋（語意 + TF-IDF）"""
        qv = self.embedding_model.encode([query]).astype("float32")
        _, sem_idxs = self.index.search(qv, top_k * 2)

        qv_tfidf = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * qv_tfidf.T).toarray().flatten()
        tfidf_idxs = tfidf_scores.argsort()[-top_k * 2:][::-1]

        combined = list(set(sem_idxs[0].tolist() + tfidf_idxs.tolist()))
        return [self.chunks[i] for i in combined[:top_k]]

    def strategy_4_reranking(self, query: str, top_k: int = 3):
        """策略4: 重新排序（LLM 評分）"""
        candidates = self.strategy_1_basic_similarity(query, top_k=top_k * 2)
        scored = []
        for chunk in candidates:
            prompt = (
                f"問題：{query}\n\n文本：{chunk[:200]}...\n\n"
                f"這段文本與問題的相關度(0-10)，只回覆數字："
            )
            try:
                resp = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=10,
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                nums = re.findall(r'\d+', raw)
                score = float(nums[0]) if nums else 0
            except Exception:
                score = 0
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def strategy_5_multi_query(self, query: str, top_k: int = 3):
        """策略5: 多查詢擴展"""
        expand_prompt = (
            f"將以下問題改寫成3個相關但不同角度的問題，用換行分隔：\n{query}"
        )
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": expand_prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            queries = [query] + resp.choices[0].message.content.strip().split('\n')[:3]
        except Exception:
            queries = [query]

        all_chunks = []
        for q in queries:
            all_chunks.extend(self.strategy_1_basic_similarity(q, top_k=2))
        return list(dict.fromkeys(all_chunks))[:top_k]

    def strategy_6_contextual_compression(self, query: str, top_k: int = 3):
        """策略6: 上下文壓縮"""
        chunks = self.strategy_1_basic_similarity(query, top_k=top_k)
        compressed = []
        for chunk in chunks:
            prompt = (
                f"從以下文本中提取與問題「{query}」最相關的1-2句話：\n\n{chunk}"
            )
            try:
                resp = self.client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0,
                )
                compressed.append(resp.choices[0].message.content.strip())
            except Exception:
                compressed.append(chunk[:300])
        return compressed

    def strategy_7_parent_child(self, query: str, top_k: int = 3):
        """策略7: 父子文檔"""
        full_text = ' '.join(self.chunks)
        small_chunks = self._split_text(full_text, chunk_size=300, overlap=50)
        small_emb = self.embedding_model.encode(
            small_chunks, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")

        small_index = faiss.IndexFlatL2(small_emb.shape[1])
        small_index.add(small_emb)

        qv = self.embedding_model.encode([query]).astype("float32")
        _, idxs = small_index.search(qv, top_k)

        results = []
        for idx in idxs[0]:
            snippet = small_chunks[idx]
            for big in self.chunks:
                if snippet in big:
                    results.append(big)
                    break
        return list(dict.fromkeys(results))[:top_k]

    def strategy_8_hypothetical_answer(self, query: str, top_k: int = 3):
        """策略8: 假設性答案（HyDE）"""
        hyde_prompt = (
            f"請對以下問題給出一個假設性的答案（即使不確定）：\n{query}"
        )
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": hyde_prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            hypothetical = resp.choices[0].message.content
        except Exception:
            hypothetical = query

        qv = self.embedding_model.encode([hypothetical]).astype("float32")
        _, idxs = self.index.search(qv, top_k)
        return [self.chunks[i] for i in idxs[0]]

    # ── 主問答入口 ──────────────────────────
    STRATEGIES = {
        "1. 基礎語意搜尋":        "strategy_1_basic_similarity",
        "2. TF-IDF 關鍵詞":      "strategy_2_tfidf",
        "3. 混合搜尋":            "strategy_3_hybrid",
        "4. 重新排序":            "strategy_4_reranking",
        "5. 多查詢擴展":          "strategy_5_multi_query",
        "6. 上下文壓縮":          "strategy_6_contextual_compression",
        "7. 父子文檔":            "strategy_7_parent_child",
        "8. 假設性答案 (HyDE)":   "strategy_8_hypothetical_answer",
    }

    def generate_answer(self, query: str, strategy: str, top_k: int = 3):
        if not self.chunks:
            return "❌ 請先上傳 PDF 檔案！", []

        method = getattr(self, self.STRATEGIES.get(strategy, "strategy_1_basic_similarity"))
        relevant_chunks = method(query, top_k)
        context = "\n\n---\n\n".join(relevant_chunks)

        prompt = (
            f"請根據以下上下文回答問題。如果上下文中沒有相關資訊，請說明無法回答。\n\n"
            f"上下文：\n{context}\n\n問題：{query}\n\n請用繁體中文詳細回答："
        )
        try:
            resp = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "你是專業的文件分析助手。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            return resp.choices[0].message.content, relevant_chunks
        except Exception as e:
            return f"❌ 生成答案失敗：{e}", []


# ─────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "load_msg" not in st.session_state:
    st.session_state.load_msg = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "last_strategy" not in st.session_state:
    st.session_state.last_strategy = ""


# ─────────────────────────────────────────────
# Sidebar — 設定
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 系統設定")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="前往 https://console.groq.com 取得免費 API Key",
    )

    st.markdown("---")
    st.markdown("## 📤 上傳 PDF")
    uploaded_file = st.file_uploader("選擇 PDF 檔案", type=["pdf"])

    if st.button("🚀 載入文件", use_container_width=True, type="primary"):
        if not api_key:
            st.error("請先輸入 Groq API Key")
        elif uploaded_file is None:
            st.warning("請先選擇 PDF 檔案")
        else:
            with st.spinner("正在解析 PDF 並建立索引…"):
                # 寫入臨時檔
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    rag = MultiStrategyRAG(api_key=api_key)
                    msg = rag.load_pdf(tmp_path)
                    if msg.startswith("✅"):
                        st.session_state.rag = rag
                        st.session_state.pdf_loaded = True
                    st.session_state.load_msg = msg
                finally:
                    os.unlink(tmp_path)

    if st.session_state.load_msg:
        if "✅" in st.session_state.load_msg:
            st.success(st.session_state.load_msg)
        else:
            st.error(st.session_state.load_msg)

    st.markdown("---")
    st.markdown("## 🎯 RAG 策略")
    strategy = st.selectbox(
        "選擇策略",
        list(MultiStrategyRAG.STRATEGIES.keys()),
        index=0,
    )

    top_k = st.slider("檢索片段數量 (Top-K)", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.markdown("""
### 📖 策略說明
| # | 名稱 | 方法 |
|---|------|------|
| 1 | 基礎語意 | 向量相似度 |
| 2 | TF-IDF | 詞頻統計 |
| 3 | 混合搜尋 | 語意＋關鍵詞 |
| 4 | 重新排序 | LLM 評分 |
| 5 | 多查詢 | 生成多角度問題 |
| 6 | 上下文壓縮 | LLM 提取摘要 |
| 7 | 父子文檔 | 小→大上下文 |
| 8 | HyDE | 先生成假設答案 |
""")


# ─────────────────────────────────────────────
# 主頁面
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤖 多策略 RAG PDF 問答系統</h1>
  <p>8 種檢索策略 × Groq Llama 3.1 × 語意向量搜尋 — 智能解析您的文件</p>
</div>
""", unsafe_allow_html=True)

# 問題輸入區
st.markdown("### 💬 提問")
col_q, col_btn = st.columns([5, 1])
with col_q:
    question = st.text_area(
        "輸入您的問題",
        placeholder="例如：這份文件的主要內容是什麼？",
        height=100,
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    ask_clicked = st.button("🔍 提問", use_container_width=True, type="primary")

# 範例問題
st.markdown("**範例問題：**")
examples = [
    "這份文件的主要內容是什麼？",
    "文件中提到哪些重要概念？",
    "有哪些關鍵數據或統計資料？",
    "文件的結論是什麼？",
]
ex_cols = st.columns(len(examples))
for col, ex in zip(ex_cols, examples):
    if col.button(ex, use_container_width=True):
        question = ex
        ask_clicked = True

st.markdown("---")

# 執行問答
if ask_clicked:
    if not question.strip():
        st.warning("⚠️ 請輸入問題")
    elif not st.session_state.pdf_loaded or st.session_state.rag is None:
        st.error("❌ 請先在左側上傳並載入 PDF 文件")
    else:
        with st.spinner(f"使用「{strategy}」策略搜尋中…"):
            answer, sources = st.session_state.rag.generate_answer(
                question, strategy, top_k
            )
        st.session_state.answer = answer
        st.session_state.sources = sources
        st.session_state.last_strategy = strategy

# 顯示答案
if st.session_state.answer:
    st.markdown("### 💡 AI 回答")
    st.markdown(
        f'<span class="badge">策略：{st.session_state.last_strategy}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="answer-box">{st.session_state.answer}</div>',
        unsafe_allow_html=True,
    )

    # 來源片段
    if st.session_state.sources:
        with st.expander(
            f"📚 查看檢索到的 {len(st.session_state.sources)} 個文本片段", expanded=False
        ):
            for i, chunk in enumerate(st.session_state.sources, 1):
                st.markdown(
                    f'<div class="source-chunk">'
                    f'<div class="chunk-label">片段 {i}</div>'
                    f'{chunk}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
