import streamlit as st
import re
import io
import numpy as np
import pandas as pd
import pdfplumber
import docx
import spacy
import datetime
from collections import defaultdict

from pathlib import Path
from kiwipiepy import Kiwi

import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Document Preprocessing",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# 경로 / 상수
# ============================================================
BASE_DIR = Path('.')
UNIT_DIR = BASE_DIR / 'units'
UNIT_DIR.mkdir(exist_ok=True)

UNIT_FILES = {
    'word':      UNIT_DIR / 'word.csv',
    'sentence':  UNIT_DIR / 'sentence.csv',
    'paragraph': UNIT_DIR / 'paragraph.csv',
    'document':  UNIT_DIR / 'document.csv',
}
INTEGRATED_CSV = UNIT_DIR / 'integrated.csv'
META_CSV       = UNIT_DIR / 'metadata.csv'

UNITS_TO_SAVE = ['word', 'sentence', 'document']   # paragraph 잠정 보류

UNIT_COLS  = ['파일명', '작성자유형', '문서유형', '원문', '토큰']
INTEG_COLS = ['파일명', '작성자유형', '문서유형', '단위', '원문', '토큰']
META_COLS  = ['파일명', '문서유형', '작성자유형', '처리일시']

SYNC_FILES = {
    'word.csv':       UNIT_FILES['word'],
    'sentence.csv':   UNIT_FILES['sentence'],
    'document.csv':   UNIT_FILES['document'],
    'integrated.csv': INTEGRATED_CSV,
    'metadata.csv':   META_CSV,
}

AUTHOR_TYPES = ['Human', 'AI']
DOC_TYPES    = ['report', 'essay']

# 감성사전 (같은 폴더에 위치) — 음식점 리뷰 기반, 임시 사용
SENTI_FILENAME = 'V4_SOPMI_감성사전_수동정제.csv'

METRIC_COLS = ['TTR', '평균 문장 길이', '문장 길이 분산', 'Burstiness', '감정 다양성']


def init_local_csvs():
    for name in ['word', 'sentence', 'document']:
        if not UNIT_FILES[name].exists():
            pd.DataFrame(columns=UNIT_COLS).to_csv(
                UNIT_FILES[name], index=False, encoding='utf-8-sig')
    if not INTEGRATED_CSV.exists():
        pd.DataFrame(columns=INTEG_COLS).to_csv(
            INTEGRATED_CSV, index=False, encoding='utf-8-sig')
    if not META_CSV.exists():
        pd.DataFrame(columns=META_COLS).to_csv(
            META_CSV, index=False, encoding='utf-8-sig')


# ============================================================
# 모델 / 감성사전 초기화 (캐싱)
# ============================================================
@st.cache_resource
def load_models():
    kiwi   = Kiwi()
    nlp_en = spacy.load('en_core_web_sm')
    kfont  = font_manager.findfont(plt.rcParams['font.family'][0])
    return kiwi, nlp_en, kfont

kiwi, nlp_en, KFONT_PATH = load_models()


@st.cache_data
def load_sentiment_dict():
    """감성사전 CSV → {token: score(-2~2)}. 없으면 None"""
    p = BASE_DIR / SENTI_FILENAME
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'token' not in df.columns or 'expert_adjusted' not in df.columns:
        return None
    sub = df[['token', 'expert_adjusted']].dropna()
    return dict(zip(sub['token'].astype(str), sub['expert_adjusted'].astype(float)))

SENTI = load_sentiment_dict()


# ============================================================
# 구글 드라이브 연동
# ============================================================
SCOPES = ['https://www.googleapis.com/auth/drive']


@st.cache_resource
def get_drive_service():
    info  = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build('drive', 'v3', credentials=creds)


def drive_find_child(service, parent_id, name, is_folder=True):
    mime = " and mimeType='application/vnd.google-apps.folder'" if is_folder else ""
    q = f"'{parent_id}' in parents and name='{name}'{mime} and trashed=false"
    res = service.files().list(
        q=q, fields="files(id, name)",
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    files = res.get('files', [])
    return files[0]['id'] if files else None


def drive_ensure_folder(service, parent_id, name):
    fid = drive_find_child(service, parent_id, name, is_folder=True)
    if fid:
        return fid
    meta = {'name': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]}
    folder = service.files().create(
        body=meta, fields='id', supportsAllDrives=True).execute()
    return folder['id']


def drive_list_pdfs(service, folder_id):
    q = (f"'{folder_id}' in parents and trashed=false "
         f"and mimeType='application/pdf'")
    res = service.files().list(
        q=q, fields="files(id, name)", pageSize=1000,
        supportsAllDrives=True, includeItemsFromAllDrives=True,
    ).execute()
    return res.get('files', [])


def drive_download_bytes(service, file_id):
    request    = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf        = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return buf.read()


def drive_upload_csv(service, folder_id, filename, local_path):
    media = MediaIoBaseUpload(
        io.BytesIO(local_path.read_bytes()),
        mimetype='text/csv', resumable=False)
    existing = drive_find_child(service, folder_id, filename, is_folder=False)
    if existing:
        service.files().update(
            fileId=existing, media_body=media,
            supportsAllDrives=True).execute()
    else:
        service.files().create(
            body={'name': filename, 'parents': [folder_id]},
            media_body=media, fields='id',
            supportsAllDrives=True).execute()


def get_folder_structure():
    service = get_drive_service()
    root_id = st.secrets["drive"]["root_folder_id"]
    folders = {'output': drive_ensure_folder(service, root_id, 'output')}
    for author in AUTHOR_TYPES:
        author_id = drive_ensure_folder(service, root_id, author)
        for doc in DOC_TYPES:
            folders[(author, doc)] = drive_ensure_folder(service, author_id, doc)
    return service, folders


def sync_from_drive(service, output_id):
    for fname, local_path in SYNC_FILES.items():
        fid = drive_find_child(service, output_id, fname, is_folder=False)
        if fid:
            local_path.write_bytes(drive_download_bytes(service, fid))
    init_local_csvs()


def sync_to_drive(service, output_id):
    for fname, local_path in SYNC_FILES.items():
        if local_path.exists():
            drive_upload_csv(service, output_id, fname, local_path)


# ============================================================
# 텍스트 추출
# ============================================================
def extract_text_from_bytes(file_bytes, suffix='.pdf'):
    if suffix == '.pdf':
        texts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                bboxes = [t.bbox for t in page.find_tables()]
                if bboxes:
                    p = page
                    for bbox in bboxes:
                        p = p.outside_bbox(bbox)
                    text = p.extract_text()
                else:
                    text = page.extract_text()
                if text:
                    texts.append(text)
        return '\n'.join(texts)

    elif suffix == '.docx':
        doc   = docx.Document(io.BytesIO(file_bytes))
        texts = []
        for block in doc.element.body:
            if block.tag.endswith('}p'):
                para = docx.text.paragraph.Paragraph(block, doc)
                if para.text.strip():
                    texts.append(para.text)
        return '\n'.join(texts)

    else:
        raise ValueError(f"지원하지 않는 형식: {suffix}")


# ============================================================
# 기본 전처리
# ============================================================
def remove_html_tags(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
    return text

def remove_urls(text):
    for p in [r'https?://\S+', r'www\.\S+',
              r'\S+\.com\S*', r'\S+\.co\.kr\S*']:
        text = re.sub(p, '', text, flags=re.IGNORECASE)
    return text

def clean_special_characters(text):
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    for char in ['\xa0', '\x00', '\r', '\ufeff']:
        text = text.replace(char, ' ')
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[『』「」]', '"', text)
    text = re.sub(r'[〔〕【】]',
                  lambda m: '[' if m.group() in '〔【' else ']', text)
    text = re.sub(r'…', '...', text)
    return text

def normalize_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def preprocess_basic(text):
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = clean_special_characters(text)
    text = normalize_whitespace(text)
    return text


# ============================================================
# 토큰화
# ============================================================
def lemmatize_english(word):
    doc = nlp_en(word)
    return doc[0].lemma_ if doc else word

def tokenize(text):
    tokens       = kiwi.tokenize(text)
    result       = []
    total        = len(tokens)
    skip_indices = set()

    for i, token in enumerate(tokens):
        if i in skip_indices:
            continue

        word = token.form
        pos  = str(token.tag)

        if pos == 'SL':
            lemma = lemmatize_english(word)
            result.append((word, pos, 'en', lemma))

        elif pos == 'SH':
            result.append((word, pos, 'zh', word))

        elif pos in ['VA', 'VV', 'VX', 'VCN', 'VCP']:
            base   = word if word.endswith('다') else word + '다'
            has_ep = False

            next_eomi     = None
            next_eomi_idx = None
            for j in range(i + 1, min(i + 4, total)):
                next_tag = str(tokens[j].tag)
                if next_tag == 'EP':
                    has_ep = True
                    continue
                if next_tag in ['EF', 'EC', 'ETM', 'ETN']:
                    next_eomi     = next_tag
                    next_eomi_idx = j
                break

            if (not has_ep
                    and next_eomi == 'EF'
                    and next_eomi_idx is not None
                    and tokens[next_eomi_idx].form == '다'
                    and not word.endswith('다')):
                skip_indices.add(next_eomi_idx)

            if next_eomi == 'EF':
                result.append((base, pos, 'ko', base))
            else:
                result.append((word, pos, 'ko', base))

        else:
            result.append((word, pos, 'ko', word))

    result = [(w, p, l, b) for w, p, l, b in result if w.strip() and b.strip()]
    return result


# ============================================================
# 다단위 분리
# ============================================================
def split_paragraphs(text, min_chars=50):
    """PDF 문단 휴리스틱 (결과 부정확 → paragraph 저장은 보류 상태)"""
    merged = re.sub(r'\n', '', text)
    merged = re.sub(r'([다요음까네]\.)\s*', r'\1\n', merged)
    sents = [s.text.strip() for s in kiwi.split_into_sents(merged)
             if s.text.strip()]
    paragraphs, buf = [], ""
    for sent in sents:
        buf = (buf + " " + sent).strip()
        if len(buf) >= min_chars:
            paragraphs.append(buf)
            buf = ""
    if buf:
        if paragraphs:
            paragraphs[-1] += " " + buf
        else:
            paragraphs.append(buf)
    return paragraphs

def split_sentences(text):
    return [s.text for s in kiwi.split_into_sents(text)]

def to_token_str(text):
    return ' '.join(t[0] for t in tokenize(preprocess_basic(text)))

def split_units(raw_text):
    units = {'word': [], 'sentence': [], 'paragraph': [], 'document': []}

    doc_clean = preprocess_basic(raw_text)
    units['document'].append((doc_clean, to_token_str(raw_text)))

    paragraphs = split_paragraphs(raw_text)
    for para in paragraphs:
        units['paragraph'].append((para, to_token_str(para)))

    for para in paragraphs:
        for sent in split_sentences(para):
            sent = sent.strip()
            if sent:
                units['sentence'].append((sent, to_token_str(sent)))

    for form, tag, lang, base in tokenize(doc_clean):
        if not form.strip() or not base.strip():
            continue
        units['word'].append((form, base))

    return units


# ============================================================
# 처리 / 저장 / 관리
# ============================================================
def process_document(fname, file_bytes, doc_type, author_type):
    try:
        raw   = extract_text_from_bytes(file_bytes, '.pdf')
        units = split_units(raw)

        for unit_name in UNITS_TO_SAVE:
            rows = [[fname, author_type, doc_type, rawtxt, tok]
                    for rawtxt, tok in units[unit_name]]
            new = pd.DataFrame(rows, columns=UNIT_COLS)
            old = pd.read_csv(UNIT_FILES[unit_name], encoding='utf-8-sig')
            pd.concat([old, new], ignore_index=True).to_csv(
                UNIT_FILES[unit_name], index=False, encoding='utf-8-sig')

        integrated_rows = []
        for unit_name in UNITS_TO_SAVE:
            for rawtxt, tok in units[unit_name]:
                integrated_rows.append(
                    [fname, author_type, doc_type, unit_name, rawtxt, tok])
        new_int = pd.DataFrame(integrated_rows, columns=INTEG_COLS)
        old_int = pd.read_csv(INTEGRATED_CSV, encoding='utf-8-sig')
        pd.concat([old_int, new_int], ignore_index=True).to_csv(
            INTEGRATED_CSV, index=False, encoding='utf-8-sig')

        meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
        new_meta = pd.DataFrame([[
            fname, doc_type, author_type,
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]], columns=META_COLS)
        pd.concat([meta, new_meta], ignore_index=True).to_csv(
            META_CSV, index=False, encoding='utf-8-sig')

        counts = {u: len(units[u]) for u in UNITS_TO_SAVE}
        return counts, None

    except Exception as e:
        return None, str(e)


def reset_all(service, output_id):
    for name in ['word', 'sentence', 'document']:
        pd.DataFrame(columns=UNIT_COLS).to_csv(
            UNIT_FILES[name], index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=INTEG_COLS).to_csv(
        INTEGRATED_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=META_COLS).to_csv(
        META_CSV, index=False, encoding='utf-8-sig')
    sync_to_drive(service, output_id)


# ============================================================
# 통계 / 그래프 — 단어 분포 · 워드클라우드
# ============================================================
def get_type_stats(df, token):
    stats = {}
    for atype in AUTHOR_TYPES:
        sub    = df[df['작성자유형'] == atype]
        total  = len(sub)
        count  = (sub['토큰'] == token).sum()
        n_docs = sub['파일명'].nunique()
        stats[atype] = {
            'count':   int(count),
            'total':   int(total),
            'ratio':   (count / total * 100) if total else 0.0,
            'n_docs':  int(n_docs),
            'per_doc': (count / n_docs) if n_docs else 0.0,
        }
    return stats


def make_wordcloud_fig(word_df):
    if word_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, atype in zip(axes, AUTHOR_TYPES):
        sub    = word_df[word_df['작성자유형'] == atype]
        tokens = sub['토큰'].dropna().astype(str)
        tokens = tokens[tokens.str.strip() != '']
        if tokens.empty:
            ax.set_title(f"{atype} (데이터 없음)")
            ax.axis('off')
            continue
        freq = tokens.value_counts().to_dict()
        wc = WordCloud(font_path=KFONT_PATH, width=700, height=500,
                       background_color='white',
                       max_words=100).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(f"{atype}  (문서 {sub['파일명'].nunique()}개)")
        ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
# Document Profiling — 파생변수 계산
# ============================================================
def sent_token_count(tok_str):
    """문장 토큰 문자열의 공백 기준 토큰 수"""
    if not isinstance(tok_str, str):
        return 0
    t = tok_str.strip()
    return len(t.split()) if t else 0


def compute_burstiness(tokens):
    """
    단어 출현 간격 기반 Burstiness = (σ-μ)/(σ+μ)  (-1:규칙적 ~ +1:편중)
    문서 내 3회 이상 출현 토큰만 사용(간격 2개 이상 확보), 그 평균 반환
    """
    pos = defaultdict(list)
    for i, t in enumerate(tokens):
        pos[t].append(i)
    bs = []
    for t, idxs in pos.items():
        if len(idxs) < 3:
            continue
        gaps = np.diff(idxs)
        mu, sd = gaps.mean(), gaps.std()
        if mu + sd > 0:
            bs.append((sd - mu) / (sd + mu))
    return float(np.mean(bs)) if bs else np.nan


def compute_doc_metrics(fname, word_df, sent_df, senti):
    """한 문서의 파생변수 5종 계산"""
    w_tokens = (word_df[word_df['파일명'] == fname]['토큰']
                .dropna().astype(str))
    w_tokens = [t for t in w_tokens if t.strip()]
    n = len(w_tokens)

    m = {}
    # 1) TTR
    m['TTR'] = (len(set(w_tokens)) / n) if n else np.nan

    # 2) 평균 문장 길이 / 3) 문장 길이 분산  (토큰 문자열 공백 기준)
    s_toks = sent_df[sent_df['파일명'] == fname]['토큰']
    lens = [sent_token_count(x) for x in s_toks]
    lens = [l for l in lens if l > 0]
    m['평균 문장 길이'] = float(np.mean(lens)) if lens else np.nan
    m['문장 길이 분산'] = float(np.var(lens)) if len(lens) >= 2 else np.nan

    # 4) Burstiness
    m['Burstiness'] = compute_burstiness(w_tokens)

    # 5) 감정 다양성 (감성사전 매칭 점수의 표준편차)
    if senti:
        scores = [senti[t] for t in w_tokens if t in senti]
        m['감정 다양성']    = float(np.std(scores)) if len(scores) >= 2 else np.nan
        m['_감성매칭수']    = len(scores)
        m['_감성매칭률(%)'] = (len(scores) / n * 100) if n else 0.0
    else:
        m['감정 다양성']    = np.nan
        m['_감성매칭수']    = 0
        m['_감성매칭률(%)'] = 0.0

    return m


def compute_all_metrics(meta, word_df, sent_df, senti):
    """전체 문서의 파생변수 표"""
    rows = []
    for _, r in meta.iterrows():
        met = compute_doc_metrics(r['파일명'], word_df, sent_df, senti)
        rows.append({
            '파일명':     r['파일명'],
            '작성자유형': r['작성자유형'],
            '문서유형':   r['문서유형'],
            **met,
        })
    return pd.DataFrame(rows)


# ============================================================
# 드라이브 초기 동기화
# ============================================================
if 'drive_ready' not in st.session_state:
    try:
        with st.spinner("구글 드라이브에서 기존 데이터를 불러오는 중..."):
            svc, fldrs = get_folder_structure()
            sync_from_drive(svc, fldrs['output'])
            st.session_state['drive_ready'] = True
            st.session_state['folders']     = fldrs
    except Exception as e:
        st.error(f"드라이브 연결 실패: {e}")
        st.info("`.streamlit/secrets.toml` 설정을 확인해주세요. (SETUP.md 참조)")
        st.stop()

service   = get_drive_service()
folders   = st.session_state['folders']
OUTPUT_ID = folders['output']
init_local_csvs()


# ============================================================
# 사이드바 — 데이터 처리 · 화면 선택 · 관리
# ============================================================
with st.sidebar:
    st.header("데이터")
    st.caption("구글 드라이브의 Human / AI 폴더에서 PDF를 읽어옵니다.")

    if st.button("드라이브 스캔", use_container_width=True):
        meta_now = pd.read_csv(META_CSV, encoding='utf-8-sig')
        done = set(meta_now['파일명']) if not meta_now.empty else set()
        pending = []
        with st.spinner("드라이브 폴더 확인 중..."):
            for author in AUTHOR_TYPES:
                for doc in DOC_TYPES:
                    fid = folders[(author, doc)]
                    for f in drive_list_pdfs(service, fid):
                        if f['name'] not in done:
                            pending.append({
                                'id': f['id'], 'name': f['name'],
                                'author': author, 'doc': doc,
                            })
        st.session_state['pending'] = pending

    pending = st.session_state.get('pending', [])
    if pending:
        st.success(f"신규 문서 {len(pending)}건")
        for p in pending[:8]:
            st.caption(f"· [{p['author']}/{p['doc']}] {p['name']}")
        if len(pending) > 8:
            st.caption(f"... 외 {len(pending) - 8}건")

        if st.button("전체 처리 시작", type="primary", use_container_width=True):
            prog = st.progress(0.0)
            ok, fail = [], []
            for i, p in enumerate(pending):
                data = drive_download_bytes(service, p['id'])
                counts, err = process_document(
                    p['name'], data, p['doc'], p['author'])
                if err:
                    fail.append((p['name'], err))
                else:
                    ok.append((p['name'], counts))
                prog.progress((i + 1) / len(pending))
            with st.spinner("드라이브에 결과 저장 중..."):
                sync_to_drive(service, OUTPUT_ID)
            for name, c in ok:
                st.success(f"{name} — word {c['word']} / "
                           f"sentence {c['sentence']} / document {c['document']}")
            for name, err in fail:
                st.error(f"{name}: {err}")
            st.session_state['pending'] = []
            st.rerun()

    st.divider()
    st.header("화면")
    menu = st.radio(
        "보기 선택",
        ["문서별 통계", "Document Profiling", "단어 분포", "워드클라우드"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("관리"):
        st.caption("모든 처리 결과 CSV를 비웁니다. (드라이브 output 포함)")
        if st.button("전체 초기화", use_container_width=True):
            reset_all(service, OUTPUT_ID)
            st.success("전체 초기화 완료")
            st.rerun()


# ============================================================
# 데이터 로드 + 상단 요약 카드 (공통)
# ============================================================
st.title("Document Preprocessing")

meta    = pd.read_csv(META_CSV, encoding='utf-8-sig')
word_df = pd.read_csv(UNIT_FILES['word'], encoding='utf-8-sig')
sent_df = pd.read_csv(UNIT_FILES['sentence'], encoding='utf-8-sig')
total   = len(meta)

c1, c2, c3 = st.columns(3)
c1.metric("총 처리 문서", f"{total}건")
if total > 0:
    c2.metric("Human 문서", f"{(meta['작성자유형'] == 'Human').sum()}건")
    c3.metric("AI 문서",    f"{(meta['작성자유형'] == 'AI').sum()}건")

st.divider()


# ============================================================
# 화면 1 — 문서별 통계
# ============================================================
if menu == "문서별 통계":
    st.subheader("문서별 통계")
    if meta.empty:
        st.info("처리된 문서가 없습니다. 사이드바에서 드라이브를 스캔하세요.")
    else:
        rows = []
        for _, r in meta.iterrows():
            fname = r['파일명']
            rows.append({
                '파일명':     fname,
                '작성자유형': r['작성자유형'],
                '문서유형':   r['문서유형'],
                '문장 수':    len(sent_df[sent_df['파일명'] == fname]),
                '단어 수':    len(word_df[word_df['파일명'] == fname]),
                '문단 수':    '',   # 잠정 보류
                '처리일시':   r['처리일시'],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ============================================================
# 화면 2 — Document Profiling
# ============================================================
elif menu == "Document Profiling":
    st.subheader("Document Profiling — 파생변수")

    if SENTI is None:
        st.warning(f"감성사전({SENTI_FILENAME})을 찾지 못해 '감정 다양성'은 계산되지 않습니다.")
    else:
        st.caption("※ 현재 감성사전은 음식점 리뷰 기반으로, 학술 문서에는 부적합할 수 있습니다. "
                   "감정 다양성 결과는 참고용입니다. (교수님 확인 예정)")

    if meta.empty or word_df.empty:
        st.info("처리된 문서가 없습니다. 사이드바에서 드라이브를 스캔하세요.")
    else:
        metrics_df = compute_all_metrics(meta, word_df, sent_df, SENTI)

        # ── 그룹 단위 비교 (주력) ──
        st.markdown("### 그룹 단위 비교 (Human vs AI)")
        st.caption("각 지표는 문서별로 계산한 뒤 그룹 평균을 낸 값입니다. "
                   "(문서 수가 달라도 공정하게 비교됩니다.)")

        group_mean = metrics_df.groupby('작성자유형')[METRIC_COLS].mean()
        for a in AUTHOR_TYPES:
            if a not in group_mean.index:
                group_mean.loc[a] = [np.nan] * len(METRIC_COLS)
        group_mean = group_mean.loc[AUTHOR_TYPES]

        # 지표별 막대그래프 (2 x 3, 마지막 칸 숨김)
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        colors = {'Human': '#4C72B0', 'AI': '#C44E52'}
        for idx, col in enumerate(METRIC_COLS):
            ax = axes[idx]
            vals = [group_mean.loc[a, col] for a in AUTHOR_TYPES]
            ax.bar(AUTHOR_TYPES, vals,
                   color=[colors[a] for a in AUTHOR_TYPES])
            ax.set_title(col)
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

        # 수치 표 (Human / AI / 차이)
        tbl = group_mean.T.copy()
        tbl.columns = [f"{c} 평균" for c in tbl.columns]
        if 'Human' in group_mean.index and 'AI' in group_mean.index:
            tbl['차이 (AI-Human)'] = group_mean.loc['AI'] - group_mean.loc['Human']
        st.dataframe(tbl.style.format("{:.4f}"), use_container_width=True)

        st.divider()

        # ── 문서 단위 조회 (보조) ──
        st.markdown("### 문서 단위 조회")
        sel = st.selectbox("문서 선택", list(metrics_df['파일명']))
        if sel:
            row = metrics_df[metrics_df['파일명'] == sel].iloc[0]
            atype = row['작성자유형']
            st.caption(f"작성자유형: **{atype}** · 문서유형: **{row['문서유형']}**")

            cols = st.columns(len(METRIC_COLS))
            for i, col in enumerate(METRIC_COLS):
                val = row[col]
                grp = group_mean.loc[atype, col] if atype in group_mean.index else np.nan
                delta = (val - grp) if (not np.isnan(val) and not np.isnan(grp)) else None
                cols[i].metric(
                    col,
                    "N/A" if np.isnan(val) else f"{val:.3f}",
                    None if delta is None else f"{delta:+.3f} vs 그룹",
                )

            if SENTI is not None:
                st.caption(f"감성사전 매칭: {int(row['_감성매칭수'])}개 "
                           f"({row['_감성매칭률(%)']:.1f}%)")


# ============================================================
# 화면 3 — 단어 분포
# ============================================================
elif menu == "단어 분포":
    st.subheader("단어 분포 조회")

    token = st.text_input("조회할 단어 (토큰)", key="dist_token")

    st.markdown("#### 전체 누적 분포")
    if token.strip() and not word_df.empty:
        stats = get_type_stats(word_df, token.strip())
        col_h, col_a = st.columns(2)
        with col_h:
            st.markdown(f"**Human** (문서 {stats['Human']['n_docs']}개)")
            st.metric("출현 빈도",    f"{stats['Human']['count']}회")
            st.metric("전체 토큰 중", f"{stats['Human']['ratio']:.2f}%")
            st.metric("문서당 평균",  f"{stats['Human']['per_doc']:.1f}회")
        with col_a:
            st.markdown(f"**AI** (문서 {stats['AI']['n_docs']}개)")
            st.metric("출현 빈도",    f"{stats['AI']['count']}회")
            st.metric("전체 토큰 중", f"{stats['AI']['ratio']:.2f}%")
            st.metric("문서당 평균",  f"{stats['AI']['per_doc']:.1f}회")
    else:
        st.info("단어를 입력하면 전체 누적 분포가 표시됩니다.")

    st.divider()
    st.markdown("#### 특정 문서 조회")
    if meta.empty:
        st.info("처리된 문서가 없습니다.")
    else:
        sel_fname = st.selectbox("조회 문서", list(meta['파일명']), key="dist_doc")
        if token.strip() and sel_fname:
            cur_df    = word_df[word_df['파일명'] == sel_fname]
            cur_total = len(cur_df)
            cur_count = (cur_df['토큰'] == token.strip()).sum()
            cur_ratio = (cur_count / cur_total * 100) if cur_total else 0.0

            m1, m2 = st.columns(2)
            m1.metric("출현 빈도",    f"{cur_count}회")
            m2.metric("전체 토큰 중", f"{cur_ratio:.2f}%")

            stats = get_type_stats(word_df, token.strip())
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            labels_a = ['선택 문서', 'Human 평균', 'AI 평균']
            values_a = [cur_ratio, stats['Human']['ratio'], stats['AI']['ratio']]
            axes[0].bar(labels_a, values_a,
                        color=['#4C72B0', '#55A868', '#C44E52'])
            axes[0].set_title(f"'{token.strip()}' 비율 비교 (%)")
            axes[0].set_ylabel('전체 토큰 중 비율 (%)')
            for i, v in enumerate(values_a):
                axes[0].text(i, v, f"{v:.2f}%", ha='center', va='bottom')

            top = cur_df['토큰'].value_counts().head(15)
            colors_b = ['#C44E52' if idx == token.strip() else '#B0B0B0'
                        for idx in top.index]
            axes[1].barh(top.index[::-1], top.values[::-1], color=colors_b[::-1])
            axes[1].set_title(f"{sel_fname}\n상위 빈도 토큰 (해당 단어 강조)")
            axes[1].set_xlabel('빈도')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("단어와 문서를 선택하면 통계와 그래프가 표시됩니다.")


# ============================================================
# 화면 4 — 워드클라우드
# ============================================================
elif menu == "워드클라우드":
    st.subheader("워드클라우드 (Human / AI)")
    fig = make_wordcloud_fig(word_df)
    if fig is None:
        st.info("처리된 문서가 없습니다.")
    else:
        st.pyplot(fig)
