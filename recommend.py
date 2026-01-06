import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional
import gspread
from google.oauth2.service_account import Credentials

# ============================================================================
# 1. 설정 및 상수
# ============================================================================
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]
SPREADSHEET_NAME = 'Hilight_db'
SHEET_USER_LOG = '사용자_로그'
SHEET_CONSULT_LOG = '상담_신청'
LOCAL_LOG_FILE = "local_log.xlsx"

# ============================================================================
# 2. 고정 태그맵 (룰베이스)
# ============================================================================
INTEREST_TAG_MAP = {
    "자녀/육아": {
        "누구": ["#우리_아이", "#자녀", "#태아", "#산모", "#가족", "#0세", "#15세", "#30세", "#만12세이하자녀"],
        "위험": ["#학교폭력", "#학교폭력피해치료", "#선천이상", "#성장단계별보장", "#일상_생활책임", "#화상", "#골절", "#질병"],
        "우선순위": ["#성장단계별보장", "#폭넓은보장", "#납입면제", "#건강관리서비스", "#어린이할인특약"],
        "변화": ["#자녀입학", "#출산예정"]
    },
    "운전": {
        "누구": ["#운전자", "#나", "#현대차_블루링크_가입고객", "#기아_커넥트_서비스_가입고객", "#제네시스_커넥티드_서비스_가입고객", "#KG_MOBILITY_인포콘_서비스_가입고객", "#르노코리아_서비스_가입고객", "#메르세데스벤츠_Mercedes_me_서비스_가입고객"],
        "위험": ["#교통사고", "#자동차사고", "#자동차사고벌금", "#변호사선임비용", "#형사합의금", "#면허정지", "#면허취소", "#자동차사고처리지원금"],
        "우선순위": ["#안전운전할인", "#블랙박스할인특약", "#커넥티드카할인특약", "#스마트안전운전UBI할인특약", "#첨단안전장치장착할인특약", "#Eco마일리지특약", "#대중교통이용할인특약"],
        "변화": ["#신차출고"]
    },
    "주택/부동산": {
        "누구": ["#주택소유자", "#다주택자", "#가족"],
        "위험": ["#화재", "#누수_화재", "#가전제품고장", "#일상_생활책임", "#배상책임", "#도난/파손"],
        "우선순위": ["#폭넓은보장", "#가성비_보험료"],
        "변화": ["#이사", "#내집마련"]
    },
    "반려동물": {
        "누구": ["#반려견", "#반려묘"],
        "위험": ["#피부질환", "#구강질환", "#슬개골", "#배상책임", "#입원치료비", "#통원치료비", "#상해", "#질병"],
        "우선순위": ["#보험료할인", "#특정처치보장", "#특정약물치료보장", "#다빈도질병보상"],
        "변화": ["#반려동물입양"]
    },
    "여행/레저": {
        "누구": ["#나", "#가족"],
        "위험": ["#상해", "#골절", "#화상", "#상해후유장해", "#배상책임", "#휴대품손해"],
        "우선순위": ["#종합보장", "#가성비_보험료"],
        "변화": []
    },
    "건강": {
        "누구": ["#나", "#부모님", "#가족", "#40세", "#60세"],
        "위험": ["#암_중증질환", "#뇌혈관질환", "#심장질환", "#허혈심장질환", "#수술_입원비", "#진단비", "#치료비", "#항암약물치료", "#방사선치료", "#전이암", "#간병인사용", "#질병", "#3대질병", "#사망", "#후유장해", "#치매"],
        "우선순위": ["#100세보장", "#간편가입", "#가성비_보험료", "#든든한_진단비", "#매년_주요치료비_지급", "#유병자도가입가능", "#종합보장", "#연금액_지급", "#노후준비"],
        "변화": ["#유병자경력", "#건강검진예정", "#노후준비"]
    }
}

# ============================================================================
# 3. 데이터 로드 및 UI 지원 함수
# ============================================================================
def load_catalog_tags():
    catalog_file = "catalog_tags.json"
    if not os.path.exists(catalog_file):
        return {"product_tags": {}, "all_tags": {}}
    try:
        with open(catalog_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"product_tags": {}, "all_tags": {}}

CATALOG_DATA = load_catalog_tags()

def get_catalog_product_tags() -> dict:
    return CATALOG_DATA.get("product_tags", {})

def get_recommended_tags_for_interest(interest: str) -> dict:
    full_tags = INTEREST_TAG_MAP.get(interest, {})
    recommended = {}
    for category, tags in full_tags.items():
        recommended[category] = tags[:4]
    return recommended

def get_all_tags_by_category(category: str) -> list:
    all_tags = set()
    for interest_tags in INTEREST_TAG_MAP.values():
        if category in interest_tags:
            all_tags.update(interest_tags[category])
    return sorted(list(all_tags))

def get_all_interests() -> list:
    return list(INTEREST_TAG_MAP.keys())

# ============================================================================
# 4. 추천 및 유사도 로직
# ============================================================================
def calculate_tag_similarity(user_tags: List[str], product_tags: List[str]) -> float:
    if not user_tags or not product_tags: return 0.0
    score = 0.0
    user_tags_set = set(user_tags)
    product_tags_set = set(product_tags)
    score += len(user_tags_set & product_tags_set) * 1.0
    for user_tag in user_tags:
        u_kw = user_tag.replace("#", "").lower()
        for p_tag in product_tags:
            if user_tag == p_tag: continue
            p_kw = p_tag.replace("#", "").lower()
            if u_kw in p_kw or p_kw in u_kw:
                score += 0.5
                break
    return score

def get_product_by_tags(selected_tags: Dict[str, List[str]]) -> Optional[str]:
    p_tags_db = CATALOG_DATA.get("product_tags", {})
    if not p_tags_db: return None
    u_tags_flat = [tag for tags in selected_tags.values() for tag in tags]
    
    best_match, best_score = None, 0.0
    for p_name, p_data in p_tags_db.items():
        p_tags_flat = []
        for tags in p_data.get("tags", {}).values(): p_tags_flat.extend(tags)
        
        sim = calculate_tag_similarity(u_tags_flat, p_tags_flat)
        risk_match = len(set(selected_tags.get("위험", [])) & set(p_data.get("tags", {}).get("위험", [])))
        final_score = sim + (risk_match * 0.5)
        
        if final_score > best_score:
            best_score, best_match = final_score, p_name
            
    return best_match if best_score >= 1.5 else None

# ============================================================================
# 5. 로컬 엑셀 저장
# ============================================================================
def _log_to_local_excel(sheet_name: str, row_data: list, columns: list):
    try:
        new_df = pd.DataFrame([row_data], columns=columns)
        
        if os.path.exists(LOCAL_LOG_FILE):
            try:
                with pd.ExcelFile(LOCAL_LOG_FILE, engine='openpyxl') as xls:
                    all_dfs = {}
                    for sn in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sn)
                        df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
                        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                        all_dfs[sn] = df
            except Exception:
                all_dfs = {}

            if sheet_name in all_dfs:
                all_dfs[sheet_name] = pd.concat([all_dfs[sheet_name], new_df], ignore_index=True)
            else:
                all_dfs[sheet_name] = new_df

            with pd.ExcelWriter(LOCAL_LOG_FILE, engine='openpyxl') as writer:
                for sn, df in all_dfs.items():
                    df.to_excel(writer, sheet_name=sn, index=False)
        else:
            new_df.to_excel(LOCAL_LOG_FILE, sheet_name=sheet_name, index=False, engine='openpyxl')
    except Exception as e:
        print(f"❌ [로컬] 기록 실패: {e}")

# ============================================================================
# 6. 구글 시트 연동 및 통합 로깅 (수정됨)
# ============================================================================
def get_sheets_client():
    try:
        if "gcp_service_account" not in st.secrets: return None
        creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception: return None

def get_or_create_sheet(client, sheet_name: str):
    if client is None: return None
    try:
        ss = client.open(SPREADSHEET_NAME)
        try: return ss.worksheet(sheet_name)
        except gspread.WorksheetNotFound: return ss.add_worksheet(title=sheet_name, rows=1000, cols=20)
    except Exception: return None

def log_user_action(visitor_id, consult_count, open_time_str, action_type, user_input="", recommended_product="", duration=0.0):
    action_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [visitor_id, consult_count, open_time_str, action_time, action_type, user_input, recommended_product, round(duration, 2)]
    headers = ['visitor_id', 'consult_count', 'open_time', 'action_time', 'action_type', 'user_input', 'recommended_product', 'duration_sec']
    
    try:
        client = get_sheets_client()
        ws = get_or_create_sheet(client, SHEET_USER_LOG)
        if ws:
            # 시트가 비어있으면 헤더 추가
            if not ws.get_all_values():
                ws.append_row(headers, value_input_option='USER_ENTERED')
            
            # [핵심 수정] value_input_option과 insert_data_option 추가
            ws.append_row(
                row, 
                value_input_option='USER_ENTERED', 
                insert_data_option='INSERT_ROWS'
            )
    except Exception as e:
        print(f"❌ [구글시트] 기록 실패: {e}")
    
    _log_to_local_excel(SHEET_USER_LOG, row, headers)

def log_consultation_request(visitor_id, consult_count, open_time_str, recommended_product, user_name="", user_phone="", user_email="", preferred_time=""):
    req_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [req_time, visitor_id, consult_count, open_time_str, recommended_product, user_name, user_phone, user_email, preferred_time, '대기중']
    headers = ['request_time', 'visitor_id', 'consult_count', 'session_start', 'recommended_product', 'name', 'phone', 'email', 'preferred_time', 'status']
    
    try:
        client = get_sheets_client()
        ws = get_or_create_sheet(client, SHEET_CONSULT_LOG)
        if ws:
            if not ws.get_all_values():
                ws.append_row(headers, value_input_option='USER_ENTERED')
            
            # [핵심 수정] 계단 현상 방지를 위해 옵션 강제 적용
            ws.append_row(
                row, 
                value_input_option='USER_ENTERED', 
                insert_data_option='INSERT_ROWS'
            )
    except Exception: pass
    
    _log_to_local_excel(SHEET_CONSULT_LOG, row, headers)
    return True

# ============================================================================
# 7. 초기화 및 외부 호출 함수
# ============================================================================
def get_recommendation(interest: str, selected_tags: Dict[str, List[str]], situation_text: str = "") -> Optional[str]:
    return get_product_by_tags(selected_tags)

def initialize_recommendation_system():
    print(f"✅ 시스템 초기화 완료 (로그: {LOCAL_LOG_FILE})")

if __name__ == "__main__":
    initialize_recommendation_system()