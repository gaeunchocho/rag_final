import streamlit as st
import os
import re
import uuid
import time
import json
from datetime import datetime
from dotenv import load_dotenv

# LangChain & Vector DB
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Recommendation System
import recommend

# ============================================================================
# 0. 벡터DB 불러오기
# ============================================================================
import streamlit as st
import gdown
import os
import zipfile

# --- 1. DB 설정 함수 (안정성 강화) ---
def setup_vector_dbs():
    base_path = os.path.dirname(os.path.abspath(__file__))
    db_configs = [
        {"id": "1ttI_cujWXDOBFkD6WO_vlI21V3YGzgSB", "zip_name": "chroma_db_catalog.zip", "folder": "chroma_db_catalog"},
        {"id": "11D34U49KZwgJLnURnCu8K4p8kKjBlaL4", "zip_name": "chroma_db_clause.zip", "folder": "chroma_db_clause"}
    ]

    needed = [db for db in db_configs if not os.path.exists(os.path.join(base_path, db["folder"]))]
    
    if not needed:
        return True

    # 데이터가 없을 때만 화면에 상태 표시
    with st.status("🚀 최초 실행을 위한 데이터베이스 구성 중...", expanded=True) as status:
        for db in needed:
            st.write(f"📥 {db['folder']} 다운로드 중 (약 30초 소요)...")
            url = f'https://drive.google.com/uc?id={db["id"]}'
            zip_path = os.path.join(base_path, db["zip_name"])
            
            try:
                gdown.download(url, zip_path, quiet=False, fuzzy=True)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(base_path)
                os.remove(zip_path)
            except Exception as e:
                st.error(f"다운로드 실패: {e}")
                return False
        status.update(label="✅ 구성 완료! 서비스를 시작합니다.", state="complete", expanded=False)
    return True

# --- 2. 메인 실행 로직 ---
def main():
    # [중요] 최상단이 아닌 여기서 실행해야 Streamlit이 Health Check에 성공합니다.
    if not setup_vector_dbs():
        st.error("데이터베이스 로드에 실패했습니다. 관리자에게 문의하세요.")
        st.stop()

    # 이후 기존 로직 (st.title, load_vectorstore 등) 진행
    st.title("💡 현대해상 Hi-light")
    # ... 나머지 코드 ...

if __name__ == "__main__":
    main()


# 이후 기존 app.py 코드 진행...
# ============================================================================
# 1. 환경 설정 및 스타일링 (주황/남색 계열 적용)
# ============================================================================
load_dotenv()

st.set_page_config(
    page_title="현대해상 Hi-light",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 모바일 앱 스타일 CSS (주황색/남색 테마 적용)
st.markdown("""
<style>
    /* Font & Base */
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
    .stApp { font-family: 'Pretendard', sans-serif; background-color: #FFF8E1; } /* 아주 연한 주황 배경 */
    
    /* Header/Footer Hide */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero Card (추천 결과) */
    .hero-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(255, 159, 67, 0.15); /* 주황색 그림자 */
        border: 1px solid #FFE0B2; /* 연한 주황 테두리 */
        border-left: 5px solid #FF9F43; /* 주황색 포인트 */
        position: relative;
        overflow: hidden;
    }
    
    .score-badge {
        position: absolute;
        top: 20px;
        right: 20px;
        background: #FFF3E0; /* 아주 연한 주황 배경 */
        color: #E65100; /* 진한 주황 텍스트 */
        font-weight: 800;
        font-size: 14px;
        padding: 6px 12px;
        border-radius: 12px;
        border: 1px solid #FFCC80;
    }

    .hero-label {
        display: inline-block;
        background: linear-gradient(90deg, #FF9F43, #FFB74D); /* 주황색 그라데이션 */
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }

    .product-title {
        color: #1A237E; /* 진한 남색 타이틀 */
        margin: 0 0 10px 0;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.3;
    }

    .summary-box {
        background-color: #FFFDE7; /* 아주 연한 노랑/주황 배경 */
        padding: 14px;
        border-radius: 12px;
        color: #37474F;
        font-size: 14px;
        line-height: 1.5;
        border-left: 4px solid #FFD54F; /* 노랑/주황 포인트 */
        margin-top: 10px;
    }
    
    /* Tag Explanation Box */
    .tag-explain-box {
        background-color: #E8EAF6; /* 연한 남색 배경 */
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
        border: 1px solid #C5CAE9;
    }
    .tag-explain-title {
        color: #283593; /* 중간 남색 */
        font-weight: 700;
        font-size: 12px;
        margin-bottom: 4px;
    }
    .tag-explain-text {
        color: #1A237E; /* 진한 남색 */
        font-size: 13px;
        line-height: 1.4;
    }
    
    /* No Result Card */
    .no-result-card {
        background-color: #FFF3E0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 2px dashed #FF9F43;
        text-align: center;
    }
    .no-result-icon { font-size: 48px; margin-bottom: 12px; }
    .no-result-title { color: #E65100; font-size: 18px; font-weight: 700; margin-bottom: 8px; }
    .no-result-text { color: #BF360C; font-size: 14px; line-height: 1.6; }
    
    /* Situation Prompt Box */
    .situation-prompt {
        background: linear-gradient(135deg, #FF9F43 0%, #FF6F00 100%); /* 주황색 그라데이션 */
        color: white;
        padding: 20px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(255, 159, 67, 0.4);
    }
    .situation-prompt h3 { margin: 0 0 8px 0; font-size: 16px; font-weight: 700; }
    .situation-prompt p { margin: 0 0 12px 0; font-size: 14px; opacity: 0.95; line-height: 1.5; }

    /* Easy Explanation Box */
    .easy-box {
        background-color: #E8EAF6;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid #C5CAE9;
    }
    .easy-label { color: #283593; font-weight: 700; font-size: 13px; margin-bottom: 6px; }
    .easy-text { color: #1A237E; font-size: 14px; line-height: 1.6; font-weight: 500; }

    /* Tag Chips (Step 1) */
    .tag-container {
        display: flex; flex-wrap: wrap; gap: 8px; background-color: white; padding: 12px;
        border-radius: 12px; border: 1px solid #FFE0B2; margin-top: 10px; min-height: 50px;
    }
    .tag-chip {
        background-color: #FFF3E0; color: #E65100; padding: 6px 12px; border-radius: 20px;
        font-size: 13px; font-weight: 600; border: 1px solid #FFCC80;
    }
    .tag-placeholder { color: #90A4AE; font-size: 13px; align-self: center; }

    /* Checkbox 스타일링 (주황색 적용) */
    .stCheckbox {
        padding: 8px 12px; background-color: #FFFFFF; border-radius: 8px; border: 1px solid #FFE0B2; transition: all 0.2s ease;
    }
    .stCheckbox:hover { background-color: #FFF8E1; border-color: #FF9F43; }
    .stCheckbox > label { font-size: 13px; font-weight: 600; color: #37474F; }
    /* 체크박스 선택 시 색상 (Streamlit 기본 테마 오버라이드 필요 - 여기선 CSS만으로는 한계가 있음) */
    
    /* Buttons (주황색 테마) */
    .stButton button[type="primary"] {
        background: linear-gradient(90deg, #FF9F43, #FF6F00) !important;
        color: white !important; border: none !important;
        box-shadow: 0 4px 10px rgba(255, 159, 67, 0.3) !important;
    }
    .stButton button[type="primary"]:hover {
        background: linear-gradient(90deg, #FF6F00, #E65100) !important;
        box-shadow: 0 6px 15px rgba(255, 159, 67, 0.4) !important;
    }
    div[data-testid="stLinkButton"] a {
        background: linear-gradient(90deg, #FF9F43, #FF6F00) !important;
        color: white !important; border: none !important;
        box-shadow: 0 4px 10px rgba(255, 159, 67, 0.3) !important;
    }

    /* Loading Text */
    .loading-text { font-size: 15px; color: #546E7A; font-weight: 500; text-align: center; margin-top: 15px; }
    
    /* Consultation Banner */
    .consultation-banner {
        background: linear-gradient(135deg, #FF9F43 0%, #FF6F00 100%);
        color: white; padding: 20px; border-radius: 16px; text-align: center; margin: 20px 0;
        box-shadow: 0 4px 15px rgba(255, 159, 67, 0.4);
    }
    .consultation-banner h3 { margin: 0 0 8px 0; font-size: 18px; font-weight: 700; }
    .consultation-banner p { margin: 0; font-size: 14px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1.5. Global Data Load
# ============================================================================
def load_toc_data():
    toc_path = "toc_meta_summary.txt"
    if os.path.exists(toc_path):
        with open(toc_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "TOC Data Not Found."

if "global_toc_data" not in st.session_state:
    st.session_state.global_toc_data = load_toc_data()

# 기존 상대 경로 대신 절대 경로 권장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_clause")
CATALOG_DIR = os.path.join(BASE_DIR, "chroma_db_catalog")
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"

# ============================================================================
# 2. Data Constants
# ============================================================================

def get_tag_hierarchy():
    """recommend.py의 추천 데이터를 TAG_HIERARCHY 형식으로 변환"""
    interests = recommend.get_all_interests()
    
    hierarchy = {
        "관심사": {},
        "누구": recommend.get_all_tags_by_category("누구"),
        "위험": recommend.get_all_tags_by_category("위험"),
        "우선순위": recommend.get_all_tags_by_category("우선순위"),
        "변화": recommend.get_all_tags_by_category("변화")
    }
    
    for interest in interests:
        hierarchy["관심사"][interest] = recommend.get_recommended_tags_for_interest(interest)
    
    return hierarchy

TAG_HIERARCHY = get_tag_hierarchy()

# UI 전용 데이터
PRODUCT_LINKS = {
    "개인용 자동차보험": "https://www.hi.co.kr/serviceAction.do?menuId=100212",
    "간편한 3.10.10 건강보험(세만기형)": "https://www.hi.co.kr/serviceAction.do?menuId=202652",
    "간편한3·10·10건강보험": "https://www.hi.co.kr/serviceAction.do?menuId=202652",
    "골든타임 수술종합보험": "https://www.hi.co.kr/serviceAction.do?menuId=204360",
    "굿앤굿스타 종합보험(세만기형)": "https://www.hi.co.kr/serviceAction.do?menuId=100223",
    "굿앤굿 어린이종합보험Q": "https://www.hi.co.kr/serviceAction.do?menuId=100222",
    "내삶엔(3N) 맞춤간편 건강보험": "https://www.hi.co.kr/serviceAction.do?menuId=203552",
    "뉴하이카 운전자상해보험": "https://www.hi.co.kr/serviceAction.do?menuId=100215",
    "굿앤굿 우리펫보험": "https://www.hi.co.kr/serviceAction.do?menuId=202403",
    "퍼펙트플러스 종합보험(세만기형)": "https://www.hi.co.kr/serviceAction.do?menuId=202211", # 링크 수정
    "행복가득 생활보장보험": "https://www.hi.co.kr/serviceAction.do?menuId=100242",
    "두배받는 암보험": "https://www.hi.co.kr/serviceAction.do?menuId=100224",
    "노후웰스보험": "https://www.hi.co.kr/serviceAction.do?menuId=100231" # 추가
}

# ============================================================================
# 3. Resource Loading
# ============================================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings, collection_name="insurance_rag")
    return None

@st.cache_resource
def load_catalog_vectorstore():
    """카탈로그 전용 벡터스토어 로드"""
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    if os.path.exists(CATALOG_DIR) and os.listdir(CATALOG_DIR):
        return Chroma(persist_directory=CATALOG_DIR, embedding_function=embeddings, collection_name="insurance_catalog")
    return None

@st.cache_resource
def get_llm():
    api_key = st.secrets["GOOGLE_API_KEY"]
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        google_api_key=api_key, 
        temperature=0
    )

# Session State 초기화
if "step" not in st.session_state: st.session_state.step = 1
if "selected_interest" not in st.session_state: st.session_state.selected_interest = None
if "selected_tags" not in st.session_state: st.session_state.selected_tags = {"누구": [], "위험": [], "우선순위": [], "변화": []}
if "natural_language_inputs" not in st.session_state: st.session_state.natural_language_inputs = {"누구": "", "위험": "", "우선순위": "", "변화": ""}
if "situation" not in st.session_state: st.session_state.situation = {"when": None, "where": None, "what": None, "text": ""}
if "catalog_result" not in st.session_state: st.session_state.catalog_result = None
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ============================================================================
# 4. Analysis Engine
# ============================================================================

def preprocess_text(text):
    """RAG 검색 결과의 가독성을 높이기 위한 전처리 함수"""
    if not text:
        return ""
    
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</br>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\|[\s-]+\|', '\n', text)
    text = text.replace('|', '  ')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# ============================================================================
# 4.1. 카탈로그 탐색 (1단계) - LLM 기반
# ============================================================================
def analyze_catalog_tags_with_llm(catalog_vectorstore, llm, tags, natural_language_inputs):
    """
    1단계: LLM 기반 카탈로그 분석
    """
    
    # catalog_tags.json 로드
    catalog_product_tags = recommend.get_catalog_product_tags()
    
    # 태그 문자열 생성 (자연어 포함)
    tag_descriptions = []
    for category, tag_list in tags.items():
        if tag_list:
            tag_descriptions.append(f"{category}: {', '.join(tag_list)}")
        
        # 자연어 입력 추가
        nl_input = natural_language_inputs.get(category, "").strip()
        if nl_input:
            tag_descriptions.append(f"{category} (자연어): {nl_input}")
    
    tag_str = " | ".join(tag_descriptions)
    
    # 약관 DB 검색 (k=5로 증가)
    retriever = catalog_vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(tag_str)
    
    def format_catalog_docs(docs):
        return "\n".join([
            f"<상품 {i+1}>\n- 상품명: {d.metadata.get('source', '알 수 없음')}\n- 설명: {preprocess_text(d.page_content)[:500]}..."
            for i, d in enumerate(docs)
        ])
    
    # catalog_tags.json을 문자열로 변환
    catalog_context = json.dumps(catalog_product_tags, ensure_ascii=False, indent=2)

    template = """당신은 보험 상품 추천 전문가입니다.
고객의 태그와 자연어 설명을 종합적으로 분석하여 최적의 상품을 추천하세요.

**[카탈로그 상품 태그 (catalog_tags.json)]**
{catalog_context}

**[약관 DB 검색 결과]**
{docs_context}

**[고객 선택 정보]**
{tags}

---
**[분석 절차]**
1. **자연어 처리**: 고객이 입력한 자연어를 분석하여 숨겨진 니즈 파악
2. **태그 유사도 계산**: catalog_tags.json의 상품 태그와 비교
3. **약관 검증**: 실제 보장 내용 확인
4. **종합 판단**: 태그 + 자연어 + 약관을 종합하여 최적 상품 선택

**[중요 원칙]**
1. 자연어 입력이 있으면 태그보다 우선시
2. 태그별로 **왜 이 상품이 적합한지 60자 이내로 설명**
3. 고객이 마주할 **실제 위험 상황 시나리오** 생성 (구체적이고 현실적으로)
4. 유사도가 낮으면 솔직하게 "없음" 처리

---
**[출력 형식 - JSON Only]**
마크다운 없이 순수 JSON만 출력하세요.

**상품이 있는 경우:**
{{
    "has_product": true,
    "product_name": "정확한 상품명 (카탈로그에 있는 이름)",
    "features": ["핵심 특약1", "핵심 특약2"],
    "tag_explanations": {{
        "#태그1": "적합한 이유 (60자 이내)",
        "#태그2": "적합한 이유 (60자 이내)",
        "(자연어입력된 경우)": "적합한 이유(60자 이내)"
    }},
    "risk_scenario": "고객이 실제로 마주칠 수 있는 구체적인 위험 상황 (100자 이내, 1인칭 시점)",
    "confidence": "high/medium/low",
    "matching_score": 85
}}

**상품이 없는 경우:**
{{
    "has_product": false,
    "reason": "적합한 상품이 없는 구체적인 이유",
    "confidence": "low",
    "matching_score": 0
}}

**중요**: 
- risk_scenario는 반드시 1인칭 시점으로 작성
- 태그 조합에서 자연스럽게 발생할 수 있는 실제 상황
- 예: "제가 사는 아파트 베란다 배관이 터져서 아랫집 천장이 물에 젖었습니다. 도배 비용을 물어줘야 하는데 보험으로 처리될까요?"
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {
            "tags": lambda x: tag_str,
            "catalog_context": lambda x: catalog_context[:4000], # 컨텍스트 길이 증가
            "docs_context": lambda x: format_catalog_docs(docs)
        }
        | prompt | llm | StrOutputParser()
    )
    return chain.stream(tag_str)

# ============================================================================
# 4.2. 상황 기반 분석 (2단계)
# ============================================================================
def analyze_tags_and_situation(vectorstore, llm, tags, situation_text):
    """Logic 2: 상황 기반 (가정법 화법 적용)"""
    
    current_toc_summary = st.session_state.get("global_toc_data", "목차 데이터 없음")
    tag_str = ", ".join([f"{k}: {', '.join(v)}" for k, v in tags.items() if v])
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.invoke(f"{situation_text} {tag_str}")
    
    def format_docs_with_meta(docs):
        return "\n".join([f"<Chunk {i+1}>\n- Metadata: {d.metadata}\n- Content: {preprocess_text(d.page_content)[:600]}..." for i, d in enumerate(docs)])

    template = """당신은 보험 소비자의 이익을 최우선으로 하는 객관적인 '보상 분석관'입니다.
상품을 판매하려 하지 말고, 약관에 의거하여 냉철하게 분석하세요.

아래 제공된 정보를 바탕으로 사용자의 상황을 정밀 분석하세요.

**[전체 목차]** {toc_summary}
**[약관 증거]** {context}
**[사용자 정보]** 상황: {situation} / 태그: {tags}

---
**[분석 프로토콜]**
1. **매핑:** 사용자의 상황이 약관의 어느 조항(Article)에 해당하는지 찾으십시오. 약관 증거 청크의 상품명과 전체목차를 교차검증하십시오.
2. **증거 발췌:** 해당 조항의 **원문 텍스트**를 그대로 발췌하십시오. (거짓 없이)
3. **한계점 식별:** 이 상품으로 해결되지 않는 **한계점(면책사항)**을 반드시 1개 이상 찾으십시오.
4. **점수 산출:** 상황과 약관의 일치도(Match Score)를 0~100점으로 산출 (보수적 기준).

---
**[최종 출력 형식 (JSON Only)]**
마크다운 없이 순수 JSON만 출력하십시오.
**주의:** 'summary' 필드는 단정적인 표현(보장됩니다) 대신 **"이 상품에 가입되어 있다면, 보장받을 가능성이 높습니다."** 라는 가정법 화법을 사용하십시오.

{{
    "product_name": "검증된 상품명",
    "feature_name": "핵심 특약명",
    "match_score": 95,
    "summary": "가정법을 사용한 보장 가능성 요약",
    "easy_explanation": "초등학생도 이해하는 쉬운 설명",
    "reasoning": "논리적 분석 내용",
    "evidence_snippet": "제N조(조항명)\n① 항 내용...\n② 항 내용...", 
    "limitations": "이 상품이 보장하지 않는 아쉬운 점 (솔직하게)",
    "checklist": ["확인할 점 1", "확인할 점 2"]
}}
"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"tags": lambda x: tag_str, "situation": lambda x: situation_text, "context": lambda x: format_docs_with_meta(docs), "toc_summary": lambda x: current_toc_summary}
        | prompt | llm | StrOutputParser()
    )
    return chain.stream(situation_text)

# ============================================================================
# 4.3. 챗봇 응답 생성
# ============================================================================
def generate_chat_response(vectorstore, llm, question, analysis_context):
    """사용자 질문에 대해 추천 상품 맥락 + 전체 약관 검색으로 답변 생성"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(question)
    
    docs_context = "\n\n".join([
        f"[약관 {i+1}]\n상품: {doc.metadata.get('source', '알 수 없음')}\n내용: {preprocess_text(doc.page_content)[:500]}..."
        for i, doc in enumerate(relevant_docs)
    ])
    
    chat_template = """당신은 현대해상 보험 전문 상담 AI입니다.

**[이전 추천 분석 결과]**
{analysis_context}

**[검색된 관련 약관]**
{docs_context}

**[사용자 질문]**
{question}

---
**[답변 원칙]**
1. 위 약관 증거에 근거하여 답변하세요.
2. 약관에 명시되지 않은 내용은 "약관에서 확인되지 않습니다"라고 솔직히 말하세요.
3. 보장 여부는 가정법("~라면 보장받을 가능성이 있습니다")을 사용하세요.
4. 구체적인 조항명이나 특약명을 언급하여 신뢰성을 높이세요.
5. 친절하고 이해하기 쉽게 설명하세요.

답변:
"""
    
    prompt = ChatPromptTemplate.from_template(chat_template)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "analysis_context": analysis_context,
        "docs_context": docs_context,
        "question": question
    })
    
    return response

# ============================================================================
# 5. UI Rendering
# ============================================================================

def render_catalog_card(data):
    """1단계 카탈로그 결과 카드 렌더링 (링크 매칭 로직 개선, 색상 변경)"""
    try:
        import html
        import re
        
        has_product = data.get("has_product", False)
        
        if not has_product:
            reason = html.escape(str(data.get("reason", "해당 상품을 찾을 수 없습니다.")))
            
            st.markdown(f"""
            <div class="no-result-card">
                <div class="no-result-icon">😔</div>
                <div class="no-result-title">아쉽게도 해당 특징을 가진 상품은 없습니다</div>
                <div class="no-result-text">{reason}<br><br>고객님의 소중한 의견을 접수했습니다.<br>더 나은 상품 개발에 참고하겠습니다.</div>
            </div>
            """, unsafe_allow_html=True)
            
            return False
        
        prod_name = html.escape(str(data.get("product_name", "추천 상품")))
        features = data.get("features", [])
        tag_explanations = data.get("tag_explanations", {})
        risk_scenario = html.escape(str(data.get("risk_scenario", "")))
        matching_score = data.get("matching_score", 0)
        
        # catalog_tags.json에서 summary 가져오기
        catalog_product_tags = recommend.get_catalog_product_tags()
        product_summary = ""
        
        prod_name_plain = str(data.get("product_name", ""))
        
        # 상품 요약 정보 찾기 (정확한 매칭 우선)
        if prod_name_plain in catalog_product_tags:
             product_summary = catalog_product_tags[prod_name_plain].get("summary", "")
        else:
            # 유사 매칭 시도
            for catalog_prod_name, catalog_data in catalog_product_tags.items():
                if (catalog_prod_name in prod_name_plain or prod_name_plain in catalog_prod_name):
                    product_summary = catalog_data.get("summary", "")
                    break
        
        product_summary_safe = html.escape(product_summary) if product_summary else ""
        features_html = ", ".join([html.escape(f) for f in features])
        
        # 1. 카드 렌더링 (색상 변경 적용)
        st.markdown(f"""
        <div class="hero-card">
            <div class="score-badge">{matching_score}% 매칭</div>
            <div class="hero-label">AI 추천 결과</div>
            <h2 class="product-title">{prod_name}</h2>
            <div style="color:#546E7A; font-size:14px; margin-bottom:12px;">
                💡 핵심 특약: <span style="color:#F57C00; font-weight:700;">{features_html}</span>
            </div>
            {f'<div style="color:#37474F; font-size:13px; margin-top:8px; padding:10px; background-color:#FFFDE7; border-radius:8px; border-left:3px solid #FFD54F;">📌 <strong>상품 소개:</strong> {product_summary_safe}</div>' if product_summary_safe else ''}
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 태그 설명 렌더링
        st.markdown("**🏷️ 선택하신 태그에 맞는 이유**")
        for tag, explanation in tag_explanations.items():
            tag_safe = html.escape(str(tag))
            exp_safe = html.escape(str(explanation))
            st.markdown(f"""
            <div class="tag-explain-box">
                <div class="tag-explain-title">{tag_safe}</div>
                <div class="tag-explain-text">{exp_safe}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 3. 상품 링크 매칭 로직 (Fuzzy Matching)
        matched_url = None
        
        if prod_name_plain in PRODUCT_LINKS:
            matched_url = PRODUCT_LINKS[prod_name_plain]
        else:
            def normalize_name(name):
                return re.sub(r'[\s·\.\(\)Q,]+', '', str(name)).lower()

            target_clean = normalize_name(prod_name_plain)
            
            for link_name, url in PRODUCT_LINKS.items():
                link_clean = normalize_name(link_name)
                if len(target_clean) > 2 and (link_clean in target_clean or target_clean in link_clean):
                    matched_url = url
                    break
        
        if matched_url:
            st.markdown("---")
            # 링크 버튼 스타일은 CSS로 적용됨
            st.link_button(
                "🔗 보험 상품 자세히 보기",
                matched_url,
                use_container_width=True,
                type="primary" # CSS에서 이 타입을 주황색 그라데이션으로 재정의
            )
            
            log_key = f"product_link_logged_{prod_name_plain}"
            if log_key not in st.session_state:
                recommend.log_user_action(
                    visitor_id=st.session_state.visitor_id,
                    consult_count=st.session_state.consult_count,
                    open_time_str=st.session_state.open_time_str,
                    action_type="product_link_shown",
                    user_input=f"상품 링크 표시: {prod_name_plain} -> {matched_url}",
                    recommended_product=prod_name_plain,
                    duration=time.time() - st.session_state.step_start_time
                )
                st.session_state[log_key] = True

        if risk_scenario:
            st.markdown(f"""
            <div class="situation-prompt">
                <h3>💭 이런 상황은 어떻게 보장될까요?</h3>
                <p>"{risk_scenario}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.suggested_situation = risk_scenario
        
        return True
        
    except Exception as e:
        st.error(f"카드 렌더링 오류: {str(e)}")
        with st.expander("🔍 디버그 정보", expanded=False):
            st.json(data)
        return False


def render_hero_card(data):
    """2단계 상세 분석 결과 카드 렌더링 (약관 원문 정제 및 색상 변경)"""
    try:
        import html
        import re
        prod_name = str(data.get("product_name", "추천 상품"))
        score = int(data.get("match_score", 0))
        feature_name = str(data.get('feature_name', '특약 정보 없음'))
        summary = str(data.get('summary', '요약 정보 없음'))
        easy_explanation = str(data.get('easy_explanation', '설명 정보 없음'))
        reasoning = str(data.get('reasoning', '근거 정보 없음'))
        evidence_raw = str(data.get("evidence_snippet", "관련 약관 원문을 찾을 수 없습니다."))
        limitations = str(data.get("limitations", "특별한 한계점이 발견되지 않았습니다."))
        checklist = data.get('checklist', [])
        
        prod_name_safe = html.escape(prod_name)
        feature_name_safe = html.escape(feature_name)
        summary_safe = html.escape(summary)
        easy_explanation_safe = html.escape(easy_explanation)
        limitations_safe = html.escape(limitations)

        # 약관 원문 정제 (제N조(조항명) 패턴 인식 및 줄바꿈)
        evidence_formatted = evidence_raw
        # 제N조(조항명) 패턴을 찾아 줄바꿈과 스타일 적용
        evidence_formatted = re.sub(r'(제\d+조\(.*?\))', r'<br><strong>\1</strong><br>', evidence_formatted)
        # ①, ② 등의 항 번호 앞에도 줄바꿈 적용 (선택 사항)
        evidence_formatted = re.sub(r'([①-⑮])', r'<br>\1', evidence_formatted)
        # 맨 앞의 불필요한 줄바꿈 제거
        evidence_formatted = re.sub(r'^<br>', '', evidence_formatted).strip()
        
        # 1. 카드 렌더링 (색상 변경 적용)
        st.markdown(f"""
        <div class="hero-card">
            <div class="score-badge">{score}% 매칭</div>
            <div class="hero-label">AI 분석 결과</div>
            <h2 class="product-title">{prod_name_safe}</h2>
            <div style="color:#546E7A; font-size:14px; margin-bottom:12px;">
                💡 <span style="color:#F57C00; font-weight:700;">{feature_name_safe}</span> 특약이 상황에 가장 적합합니다.
            </div>
            <div class="summary-box">
                {summary_safe}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. 약관 원문 표시 (기본 닫힘 상태, 정제된 포맷 적용)
        with st.expander("📜 분석 근거: 약관 원문 보기", expanded=False):
            st.markdown(f"""
            <div style="background-color:#FFFDE7; padding:15px; border-radius:8px; border:1px dashed #FFB74D; font-size:13px; color:#37474F; line-height:1.6;">
                {evidence_formatted}
            </div>
            <p style="font-size:12px; color:#90A4AE; margin-top:5px; text-align:right;">
                * 위 내용은 현대해상 실제 약관 데이터에 기반합니다.
            </p>
            """, unsafe_allow_html=True)
            
        c1, c2 = st.columns(2)
        with c1:
            # st.info 대신 스타일 적용된 박스 사용
            st.markdown(f"""
            <div class="easy-box">
                <div class="easy-label">👶 3초 요약</div>
                <div class="easy-text">{easy_explanation_safe}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
             # st.warning 대신 스타일 적용된 박스 사용 (색상만 다르게)
            st.markdown(f"""
            <div class="easy-box" style="background-color: #FFF3E0; border-color: #FFCC80;">
                <div class="easy-label" style="color: #E65100;">⚠️ 유의할 점</div>
                <div class="easy-text" style="color: #BF360C;">{limitations_safe}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🔍 논리적 분석 내용 보기", expanded=False):
            st.write(reasoning)
            
        with st.expander("✅ 가입/청구 전 체크리스트", expanded=False):
            if checklist and isinstance(checklist, list):
                for i, item in enumerate(checklist):
                    st.checkbox(str(item), key=f"chk_{i}_{hash(str(item))}")
            else:
                st.info("체크리스트 항목이 없습니다.")
                
    except Exception as e:
        st.error(f"카드 렌더링 오류: {str(e)}")
        with st.expander("🔍 디버그 정보 (개발자용)", expanded=False):
            st.json(data)

# ============================================================================
# 6. Main App Flow
# ============================================================================

vectorstore = load_vectorstore()
catalog_vectorstore = load_catalog_vectorstore()

if not vectorstore:
    st.error("❌ 'chroma_db_clause' 폴더를 찾을 수 없습니다.")
    st.stop()

if not catalog_vectorstore:
    st.warning("⚠️ 'chroma_db_catalog' 폴더를 찾을 수 없습니다. 카탈로그 검색이 비활성화됩니다.")

llm = get_llm()

if "recommend_initialized" not in st.session_state:
    recommend.initialize_recommendation_system()
    st.session_state.recommend_initialized = True

# 세션 추적 변수
if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())
if "consult_count" not in st.session_state:
    st.session_state.consult_count = 0
if "open_time_str" not in st.session_state:
    st.session_state.open_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if "step_start_time" not in st.session_state:
    st.session_state.step_start_time = time.time()

# --- Step 1: Interest & Tag Selection (checkbox 사용) ---
if st.session_state.step == 1:
    st.title("Hi-light")
    st.caption("내 상황에 딱 맞는 보험 찾기")
    
    st.subheader("관심사를 선택해주세요")
    cols = st.columns(3)
    interests = list(TAG_HIERARCHY["관심사"].keys())
    for i, interest in enumerate(interests):
        with cols[i % 3]:
            is_selected = (st.session_state.selected_interest == interest)
            # 버튼 스타일은 CSS로 적용됨 (type="primary" 또는 기본)
            if st.button(interest, key=f"int_{i}", use_container_width=True, type="primary" if is_selected else "secondary"):
                st.session_state.selected_interest = None if is_selected else interest
                
                recommend.log_user_action(
                    visitor_id=st.session_state.visitor_id,
                    consult_count=st.session_state.consult_count,
                    open_time_str=st.session_state.open_time_str,
                    action_type="interest_select",
                    user_input=interest if not is_selected else f"deselect: {interest}",
                    recommended_product="",
                    duration=time.time() - st.session_state.step_start_time
                )
                
                st.rerun()
    
    if st.session_state.selected_interest:
        st.markdown("---")
        
        # 관심사 선택 시 고정된 태그 리스트 표시
        recommended = TAG_HIERARCHY["관심사"][st.session_state.selected_interest]
        all_tags_by_category = {
            "누구": TAG_HIERARCHY["누구"],
            "위험": TAG_HIERARCHY["위험"],
            "우선순위": TAG_HIERARCHY["우선순위"]
        }
        
        for category in ["누구", "위험", "우선순위"]:
            st.markdown(f"**{category}**")
            
            # 추천 태그 우선 표시 + 나머지 태그
            recommended_tags = recommended.get(category, [])
            other_tags = [t for t in all_tags_by_category[category] if t not in recommended_tags]
            all_tags = recommended_tags + other_tags
            
            # 최대 5개 + 자연어 입력
            display_tags = all_tags[:5]
            
            # checkbox 사용 (즉시 반응)
            cols = st.columns(3)
            for i, tag in enumerate(display_tags):
                with cols[i % 3]:
                    is_checked = tag in st.session_state.selected_tags[category]
                    
                    checked = st.checkbox(
                        tag,
                        value=is_checked,
                        key=f"chk_{category}_{i}"
                    )
                    
                    # 상태 변경 시 session_state 업데이트
                    if checked and not is_checked:
                        st.session_state.selected_tags[category].append(tag)
                        
                        recommend.log_user_action(
                            visitor_id=st.session_state.visitor_id,
                            consult_count=st.session_state.consult_count,
                            open_time_str=st.session_state.open_time_str,
                            action_type="tag_select",
                            user_input=f"{category}: {tag}",
                            recommended_product="",
                            duration=time.time() - st.session_state.step_start_time
                        )
                    elif not checked and is_checked:
                        st.session_state.selected_tags[category].remove(tag)
                        
                        recommend.log_user_action(
                            visitor_id=st.session_state.visitor_id,
                            consult_count=st.session_state.consult_count,
                            open_time_str=st.session_state.open_time_str,
                            action_type="tag_deselect",
                            user_input=f"{category}: {tag}",
                            recommended_product="",
                            duration=time.time() - st.session_state.step_start_time
                        )
            
            # 자연어 입력
            nl_key = f"nl_{category}"
            nl_input = st.text_input(
                f"💬 {category} 직접 입력",
                value=st.session_state.natural_language_inputs.get(category, ""),
                placeholder=f"편하게 말씀해주세요!",
                key=nl_key
            )
            st.session_state.natural_language_inputs[category] = nl_input

    # 선택된 태그 미리보기
    st.markdown("---")
    st.markdown("**🔖 선택된 태그**")
    
    all_selected = []
    for cat in st.session_state.selected_tags:
        all_selected.extend(st.session_state.selected_tags[cat])
    
    # 자연어 입력도 표시
    nl_texts = [f"💬 {v}" for v in st.session_state.natural_language_inputs.values() if v.strip()]
    
    if all_selected or nl_texts:
        chips_html = "".join([f'<span class="tag-chip">{t}</span>' for t in all_selected + nl_texts])
        st.markdown(f"""
        <div class="tag-container">
            {chips_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""<div class="tag-container"><span class="tag-placeholder">선택된 태그가 여기에 표시됩니다</span></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    
    has_any_input = bool(all_selected) or any(v.strip() for v in st.session_state.natural_language_inputs.values())
    
    # 버튼 스타일은 CSS로 적용됨
    if st.button("상품 찾기 🔍", type="primary", disabled=not has_any_input, use_container_width=True):
        st.session_state.step = 1.5
        st.session_state.step_start_time = time.time()
        st.rerun()

# --- Step 1.5: Catalog Search Result (LLM 기반) ---
elif st.session_state.step == 1.5:
    if not st.session_state.catalog_result:
        loading = st.empty()
        with loading.container():
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner(""):
                status = st.markdown('<p class="loading-text">📚 카탈로그에서 상품 찾는 중...</p>', unsafe_allow_html=True)
                
                stream = analyze_catalog_tags_with_llm(
                    catalog_vectorstore, 
                    llm, 
                    st.session_state.selected_tags,
                    st.session_state.natural_language_inputs
                )
                
                time.sleep(1)
                status.markdown('<p class="loading-text">🤖 AI가 태그를 분석하는 중...</p>', unsafe_allow_html=True)
                
                full_res = ""
                for chunk in stream:
                    full_res += chunk
                
                status.markdown('<p class="loading-text">✨ 분석 완료!</p>', unsafe_allow_html=True)
                time.sleep(0.5)
                
                st.session_state.catalog_result = full_res
                
                # 로그 기록 (자연어 포함)
                log_input = ", ".join([f"{k}: {', '.join(v)}" for k, v in st.session_state.selected_tags.items() if v])
                nl_log = " | ".join([f"{k}(자연어): {v}" for k, v in st.session_state.natural_language_inputs.items() if v.strip()])
                if nl_log:
                    log_input += f" | {nl_log}"
                
                recommend.log_user_action(
                    visitor_id=st.session_state.visitor_id,
                    consult_count=st.session_state.consult_count,
                    open_time_str=st.session_state.open_time_str,
                    action_type="catalog_search_with_nlp",
                    user_input=log_input,
                    recommended_product="",
                    duration=time.time() - st.session_state.step_start_time
                )
                
        loading.empty()
        st.rerun()

    try:
        json_str = st.session_state.catalog_result.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        
        has_product = render_catalog_card(data)
        
        st.markdown("---")
        
        if has_product and "suggested_situation" in st.session_state:
            # 버튼 스타일은 CSS로 적용됨
            if st.button("💬 이 상황, 자세히 알아보기", use_container_width=True, type="primary"):
                st.session_state.situation["text"] = st.session_state.suggested_situation
                st.session_state.step = 3
                st.session_state.step_start_time = time.time()
                
                recommend.log_user_action(
                    visitor_id=st.session_state.visitor_id,
                    consult_count=st.session_state.consult_count,
                    open_time_str=st.session_state.open_time_str,
                    action_type="situation_explore_auto",
                    user_input=st.session_state.suggested_situation,
                    recommended_product="",
                    duration=time.time() - st.session_state.step_start_time
                )
                
                st.rerun()
        
        if st.button("✍️ 직접 상황 입력하기", use_container_width=True):
            st.session_state.step = 2
            st.session_state.step_start_time = time.time()
            st.rerun()
        
        if st.button("⬅️ 처음으로", use_container_width=True):
            visitor_id_backup = st.session_state.visitor_id
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.visitor_id = visitor_id_backup
            st.session_state.open_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.step_start_time = time.time()
            st.rerun()
            
    except json.JSONDecodeError as e:
        st.error("❌ 분석 결과 형식 오류")
        with st.expander("🔍 상세 오류 정보", expanded=False):
            st.code(f"JSON 파싱 오류: {str(e)}\n\n원본 데이터:\n{st.session_state.catalog_result}", language="text")

# --- Step 2: Situation Input ---
elif st.session_state.step == 2:
    st.subheader("어떤 상황인가요?")
    
    user_input = st.text_area("상황을 자유롭게 적어주세요", value=st.session_state.situation["text"], height=150, placeholder="예: 주말에 축구하다가 다리가 부러졌어요.")
    st.session_state.situation["text"] = user_input
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    if c1.button("⬅️ 이전"): 
        st.session_state.step = 1.5
        st.session_state.step_start_time = time.time()
        st.rerun()
    
    # 버튼 스타일은 CSS로 적용됨
    if c2.button("분석 시작 🔍", type="primary", disabled=not user_input.strip()):
        recommend.log_user_action(
            visitor_id=st.session_state.visitor_id,
            consult_count=st.session_state.consult_count,
            open_time_str=st.session_state.open_time_str,
            action_type="situation_input_manual",
            user_input=user_input,
            recommended_product="",
            duration=time.time() - st.session_state.step_start_time
        )
        
        st.session_state.step = 3
        st.session_state.step_start_time = time.time()
        st.rerun()

# --- Step 3: Deep Analysis & Chat ---
elif st.session_state.step == 3:
    if not st.session_state.analysis_result:
        loading = st.empty()
        with loading.container():
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner(""):
                status = st.markdown('<p class="loading-text">📚 약관 책장에서 관련 페이지 찾는 중...</p>', unsafe_allow_html=True)
                
                recommended_product_name = recommend.get_recommendation(
                    interest=st.session_state.selected_interest or "",
                    selected_tags=st.session_state.selected_tags,
                    situation_text=st.session_state.situation["text"]
                )
                
                st.session_state.recommended_product_name = recommended_product_name or "알 수 없음"
                
                stream = analyze_tags_and_situation(vectorstore, llm, st.session_state.selected_tags, st.session_state.situation["text"])
                
                time.sleep(1)
                status.markdown('<p class="loading-text">🖍️ 보장 범위에 형광펜 칠하는 중...</p>', unsafe_allow_html=True)
                
                full_res = ""
                for chunk in stream:
                    full_res += chunk
                
                status.markdown('<p class="loading-text">✨ 분석 완료!</p>', unsafe_allow_html=True)
                time.sleep(0.5)
                
                st.session_state.analysis_result = full_res
                
                recommend.log_user_action(
                    visitor_id=st.session_state.visitor_id,
                    consult_count=st.session_state.consult_count,
                    open_time_str=st.session_state.open_time_str,
                    action_type="deep_analysis_complete",
                    user_input=st.session_state.situation["text"],
                    recommended_product=st.session_state.recommended_product_name,
                    duration=time.time() - st.session_state.step_start_time
                )
                
                st.session_state.consult_count += 1
                
        loading.empty()
        st.rerun()

    try:
        json_str = st.session_state.analysis_result.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        
        if "recommended_product_name" not in st.session_state:
            st.session_state.recommended_product_name = data.get("product_name", "알 수 없음")
        
        render_hero_card(data)
        
        st.markdown("---")
        
        if "consultation_submitted" not in st.session_state:
            st.session_state.consultation_submitted = False
        
        if not st.session_state.consultation_submitted:
            st.markdown("""
            <div class="consultation-banner">
                <h3>📞 전문 상담사와 1:1 상담하기</h3>
                <p>클릭 한 번으로 상담 신청 완료! 24시간 내 연락드립니다.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                # 버튼 스타일은 CSS로 적용됨
                if st.button("📞 바로 상담 신청하기", use_container_width=True, type="primary", key="quick_consult"):
                    try:
                        user_name = f"고객_{st.session_state.visitor_id[:8]}"
                        user_phone = "연락처 미입력"
                        user_email = "이메일 미입력"
                        
                        product_name = st.session_state.get("recommended_product_name", "알 수 없음")
                        
                        success = recommend.log_consultation_request(
                            visitor_id=st.session_state.visitor_id,
                            consult_count=st.session_state.consult_count,
                            open_time_str=st.session_state.open_time_str,
                            recommended_product=product_name,
                            user_name=user_name,
                            user_phone=user_phone,
                            user_email=user_email,
                            preferred_time="언제든지 가능"
                        )
                        
                        recommend.log_user_action(
                            visitor_id=st.session_state.visitor_id,
                            consult_count=st.session_state.consult_count,
                            open_time_str=st.session_state.open_time_str,
                            action_type="consultation_request_quick",
                            user_input=f"원클릭 상담 신청: {user_name}",
                            recommended_product=product_name,
                            duration=time.time() - st.session_state.step_start_time
                        )
                        
                        if success or success is None:
                            st.session_state.consultation_submitted = True
                            st.rerun()
                        else:
                            st.error("상담 신청 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                            
                    except Exception as e:
                        st.error(f"상담 신청 오류: {str(e)}")
                        st.info("💡 아래 AI 상담사에게 연락처를 남겨주시면 빠른 상담이 가능합니다.")
        
        else:
            st.success("✅ 상담 신청이 완료되었습니다!")
            st.info(f"""
            **📌 다음 단계**
            - 방문자 ID: `{st.session_state.visitor_id[:16]}...`
            - 추천 상품: **{st.session_state.get('recommended_product_name', '알 수 없음')}**
            - 영업일 기준 24시간 내에 전문 상담사가 연락드립니다.
            - 상담 전 궁금한 점은 아래 AI 상담사에게 물어보세요.
            
            💡 **Tip**: 정확한 상담을 위해 챗봇에 연락처를 남겨주시면 더 빠른 연락이 가능합니다!
            """)
        
    except json.JSONDecodeError as e:
        st.error("❌ 분석 결과 형식 오류")
        with st.expander("🔍 상세 오류 정보", expanded=False):
            st.code(f"JSON 파싱 오류: {str(e)}\n\n원본 데이터:\n{st.session_state.analysis_result}", language="text")
            
    except Exception as e:
        st.error(f"분석 결과를 표시하는 데 문제가 발생했습니다: {str(e)}")
        with st.expander("🔍 상세 오류 정보", expanded=False):
            st.code(st.session_state.analysis_result, language="text")

    st.markdown("---")
    
    st.subheader("💬 AI 상담사")
    st.caption("추천 상품뿐만 아니라 모든 약관 정보를 검색하여 답변드립니다.")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("추가로 궁금한 점을 물어보세요!"):
        recommend.log_user_action(
            visitor_id=st.session_state.visitor_id,
            consult_count=st.session_state.consult_count,
            open_time_str=st.session_state.open_time_str,
            action_type="chat_question",
            user_input=prompt,
            recommended_product="",
            duration=time.time() - st.session_state.step_start_time
        )
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("약관을 검색하여 답변을 준비하고 있습니다..."):
                response = generate_chat_response(
                    vectorstore=vectorstore,
                    llm=llm,
                    question=prompt,
                    analysis_context=st.session_state.analysis_result
                )
                st.markdown(response)
                
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 처음으로 돌아가기", use_container_width=True):
        recommend.log_user_action(
            visitor_id=st.session_state.visitor_id,
            consult_count=st.session_state.consult_count,
            open_time_str=st.session_state.open_time_str,
            action_type="session_end",
            user_input="",
            recommended_product="",
            duration=time.time() - st.session_state.step_start_time
        )
        
        visitor_id_backup = st.session_state.visitor_id
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.session_state.visitor_id = visitor_id_backup
        st.session_state.open_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.step_start_time = time.time()
        st.session_state.consultation_submitted = False
        st.rerun()

# ============================================================================
# 7. 공통 푸터 (면책 조항) - 모든 페이지 하단에 표시
# ============================================================================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='background-color: #FFFDE7; padding: 20px; border-radius: 10px; color: #546E7A; font-size: 12px; line-height: 1.6; border: 1px solid #FFE0B2;'>
    <strong>[면책 조항 및 유의사항]</strong><br>
    <ul>
        <li>본 서비스는 인공지능(AI) 기술을 활용하여 보험 약관 및 상품 설명서 데이터를 기반으로 정보를 제공하는 참고용 서비스입니다.</li>
        <li>제공되는 추천 결과 및 분석 내용은 보험 모집을 위한 법적 효력이 있는 청약 권유가 아니며, 실제 가입 가능 여부나 보장 내용은 개인의 조건에 따라 달라질 수 있습니다.</li>
        <li>AI의 답변은 부정확하거나 시의성이 떨어질 수 있으므로, 정확한 내용은 반드시 <strong>현대해상 공식 약관 및 상품 설명서</strong>를 확인하시거나 전문 상담사와 상의하시기 바랍니다.</li>
        <li>본 서비스의 결과만을 신뢰하여 발생한 손해에 대해서는 회사가 책임을 지지 않습니다.</li>
    </ul>
    <div style='text-align: center; margin-top: 10px; color: #90A4AE;'>
        &copy; 2026 현대해상 Hi-light AI Service. All rights reserved.
    </div>
</div>
""", unsafe_allow_html=True)
