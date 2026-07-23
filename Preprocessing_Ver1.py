import streamlit as st
import re
import pandas as pd
import pdfplumber
import docx
import spacy
import datetime
import csv

from pathlib import Path
from kiwipiepy import Kiwi

import koreanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="문서 전처리 시스템",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# 경로 설정
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
META_CSV       = BASE_DIR / 'metadata.csv'

UNITS_TO_SAVE = ['word', 'sentence', 'document']   # paragraph 잠정 보류

if not META_CSV.exists():
    pd.DataFrame(columns=['파일명', '문서유형', '작성자유형', '처리일시']
                ).to_csv(META_CSV, index=False, encoding='utf-8-sig')

for path in UNIT_FILES.values():
    if not path.exists():
        pd.DataFrame(columns=['파일명', '작성자유형', '문서유형', '원문', '토큰']
                    ).to_csv(path, index=False, encoding='utf-8-sig')

if not INTEGRATED_CSV.exists():
    pd.DataFrame(columns=['파일명', '작성자유형', '문서유형', '단위', '원문', '토큰']
                ).to_csv(INTEGRATED_CSV, index=False, encoding='utf-8-sig')


# ============================================================
# 모델 초기화 (캐싱)
# ============================================================
@st.cache_resource
def load_models():
    kiwi   = Kiwi()
    nlp_en = spacy.load('en_core_web_sm')
    kfont  = font_manager.findfont(plt.rcParams['font.family'][0])
    return kiwi, nlp_en, kfont

kiwi, nlp_en, KFONT_PATH = load_models()


# ============================================================
# 텍스트 추출
# ============================================================
def extract_text(file_path):
    path = Path(file_path)
    ext  = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {file_path}")

    if ext == '.pdf':
        texts = []
        with pdfplumber.open(path) as pdf:
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

    elif ext == '.docx':
        doc   = docx.Document(path)
        texts = []
        for block in doc.element.body:
            if block.tag.endswith('}p'):
                para = docx.text.paragraph.Paragraph(block, doc)
                if para.text.strip():
                    texts.append(para.text)
        return '\n'.join(texts)

    elif ext == '.txt':
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return path.read_text(encoding='cp949')

    else:
        raise ValueError(f"지원하지 않는 형식: {ext}")


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
    """PDF 문단 휴리스틱 (결과 이상함 -> paragraph 저장은 보류 상태)"""
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
# 처리 / 저장 / 삭제
# ============================================================
def process_document(fname, file_bytes, doc_type, author_type):
    tmp_path = BASE_DIR / fname
    tmp_path.write_bytes(file_bytes)

    try:
        raw   = extract_text(str(tmp_path))
        units = split_units(raw)

        for unit_name in UNITS_TO_SAVE:
            rows = [[fname, author_type, doc_type, rawtxt, tok]
                    for rawtxt, tok in units[unit_name]]
            new = pd.DataFrame(rows,
                columns=['파일명', '작성자유형', '문서유형', '원문', '토큰'])
            old = pd.read_csv(UNIT_FILES[unit_name], encoding='utf-8-sig')
            pd.concat([old, new], ignore_index=True
                     ).to_csv(UNIT_FILES[unit_name],
                              index=False, encoding='utf-8-sig')

        integrated_rows = []
        for unit_name in UNITS_TO_SAVE:
            for rawtxt, tok in units[unit_name]:
                integrated_rows.append(
                    [fname, author_type, doc_type, unit_name, rawtxt, tok])
        new_int = pd.DataFrame(integrated_rows,
            columns=['파일명', '작성자유형', '문서유형', '단위', '원문', '토큰'])
        old_int = pd.read_csv(INTEGRATED_CSV, encoding='utf-8-sig')
        pd.concat([old_int, new_int], ignore_index=True
                 ).to_csv(INTEGRATED_CSV, index=False, encoding='utf-8-sig')

        meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
        new_meta = pd.DataFrame([[
            fname, doc_type, author_type,
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]], columns=['파일명', '문서유형', '작성자유형', '처리일시'])
        pd.concat([meta, new_meta], ignore_index=True
                 ).to_csv(META_CSV, index=False, encoding='utf-8-sig')

        counts = {u: len(units[u]) for u in UNITS_TO_SAVE}
        return counts, None

    except Exception as e:
        return None, str(e)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def delete_last():
    meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
    if meta.empty:
        return None
    last_fname = meta.iloc[-1]['파일명']
    for path in list(UNIT_FILES.values()) + [INTEGRATED_CSV]:
        df = pd.read_csv(path, encoding='utf-8-sig')
        df[df['파일명'] != last_fname].to_csv(
            path, index=False, encoding='utf-8-sig')
    meta.iloc[:-1].to_csv(META_CSV, index=False, encoding='utf-8-sig')
    return last_fname


def reset_all():
    for path in UNIT_FILES.values():
        pd.DataFrame(columns=['파일명', '작성자유형', '문서유형', '원문', '토큰']
                    ).to_csv(path, index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=['파일명', '작성자유형', '문서유형', '단위', '원문', '토큰']
                ).to_csv(INTEGRATED_CSV, index=False, encoding='utf-8-sig')
    pd.DataFrame(columns=['파일명', '문서유형', '작성자유형', '처리일시']
                ).to_csv(META_CSV, index=False, encoding='utf-8-sig')


# ============================================================
# 통계 / 그래프
# ============================================================
def get_type_stats(df, token):
    stats = {}
    for atype in ['human', 'AI']:
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


def make_wordcloud_fig():
    df = pd.read_csv(UNIT_FILES['word'], encoding='utf-8-sig')
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, atype in zip(axes, ['human', 'AI']):
        sub    = df[df['작성자유형'] == atype]
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
# 사이드바 — 입력 및 관리
# ============================================================
with st.sidebar:
    st.header("문서 입력")

    uploaded_files = st.file_uploader(
        "PDF 파일 (여러 개 가능)", type="pdf", accept_multiple_files=True)
    doc_type    = st.selectbox("문서 유형", ["report", "essay"])
    author_type = st.selectbox("작성자 유형", ["human", "AI"])

    if st.button("처리 시작", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("PDF 파일을 먼저 업로드해주세요.")
        else:
            prog = st.progress(0.0)
            done = []
            for i, uf in enumerate(uploaded_files):
                counts, err = process_document(
                    uf.name, uf.read(), doc_type, author_type)
                if err:
                    st.error(f"{uf.name}: {err}")
                else:
                    done.append((uf.name, counts))
                prog.progress((i + 1) / len(uploaded_files))
            for name, c in done:
                st.success(
                    f"{name} — word {c['word']} / "
                    f"sentence {c['sentence']} / document {c['document']}")

    st.divider()
    st.header("관리")

    if st.button("직전 문서 삭제", use_container_width=True):
        removed = delete_last()
        if removed:
            st.success(f"삭제: {removed}")
        else:
            st.warning("삭제할 문서 없음")

    if st.button("전체 초기화", use_container_width=True):
        reset_all()
        st.success("전체 초기화 완료")


# ============================================================
# 메인 — 대시보드 요약
# ============================================================
st.title("문서 전처리")

meta = pd.read_csv(META_CSV, encoding='utf-8-sig')
total = len(meta)

c1, c2, c3 = st.columns(3)
c1.metric("총 처리 문서", f"{total}건")

if total > 0:
    human_n = (meta['작성자유형'] == 'human').sum()
    ai_n    = (meta['작성자유형'] == 'AI').sum()
    c2.metric("human 문서", f"{human_n}건")
    c3.metric("AI 문서", f"{ai_n}건")


# ============================================================
# 메인 — 탭
# ============================================================
tab1, tab2, tab3 = st.tabs(["단어 분포", "워드클라우드", "문서별 통계"])


# ── 탭 1: 단어 분포 ──
with tab1:
    st.subheader("단어 분포 조회")

    token = st.text_input("조회할 단어 (토큰)", key="dist_token")

    word_df = pd.read_csv(UNIT_FILES['word'], encoding='utf-8-sig')

    st.markdown("#### 전체 누적 분포")
    if token.strip() and not word_df.empty:
        stats = get_type_stats(word_df, token.strip())
        col_h, col_a = st.columns(2)
        with col_h:
            st.markdown(f"**human** (문서 {stats['human']['n_docs']}개)")
            st.metric("출현 빈도", f"{stats['human']['count']}회")
            st.metric("전체 토큰 중", f"{stats['human']['ratio']:.2f}%")
            st.metric("문서당 평균", f"{stats['human']['per_doc']:.1f}회")
        with col_a:
            st.markdown(f"**AI** (문서 {stats['AI']['n_docs']}개)")
            st.metric("출현 빈도", f"{stats['AI']['count']}회")
            st.metric("전체 토큰 중", f"{stats['AI']['ratio']:.2f}%")
            st.metric("문서당 평균", f"{stats['AI']['per_doc']:.1f}회")
    else:
        st.info("단어를 입력하면 전체 누적 분포가 표시됩니다.")

    st.divider()
    st.markdown("#### 특정 문서 조회")

    if meta.empty:
        st.info("처리된 문서가 없습니다.")
    else:
        sel_fname = st.selectbox("조회 문서", list(meta['파일명']),
                                 key="dist_doc")
        if token.strip() and sel_fname:
            cur_df    = word_df[word_df['파일명'] == sel_fname]
            cur_total = len(cur_df)
            cur_count = (cur_df['토큰'] == token.strip()).sum()
            cur_ratio = (cur_count / cur_total * 100) if cur_total else 0.0

            m1, m2 = st.columns(2)
            m1.metric("출현 빈도", f"{cur_count}회")
            m2.metric("전체 토큰 중", f"{cur_ratio:.2f}%")

            stats = get_type_stats(word_df, token.strip())

            # 그래프 A + B
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            labels_a = ['선택 문서', 'human 평균', 'AI 평균']
            values_a = [cur_ratio, stats['human']['ratio'],
                        stats['AI']['ratio']]
            axes[0].bar(labels_a, values_a,
                        color=['#4C72B0', '#55A868', '#C44E52'])
            axes[0].set_title(f"'{token.strip()}' 비율 비교 (%)")
            axes[0].set_ylabel('전체 토큰 중 비율 (%)')
            for i, v in enumerate(values_a):
                axes[0].text(i, v, f"{v:.2f}%", ha='center', va='bottom')

            top = cur_df['토큰'].value_counts().head(15)
            colors_b = ['#C44E52' if idx == token.strip() else '#B0B0B0'
                        for idx in top.index]
            axes[1].barh(top.index[::-1], top.values[::-1],
                         color=colors_b[::-1])
            axes[1].set_title(f"{sel_fname}\n상위 빈도 토큰 (해당 단어 강조)")
            axes[1].set_xlabel('빈도')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("단어와 문서를 선택하면 통계와 그래프가 표시됩니다.")


# ── 탭 2: 워드클라우드 ──
with tab2:
    st.subheader("워드클라우드 (human / AI)")
    fig = make_wordcloud_fig()
    if fig is None:
        st.info("처리된 문서가 없습니다.")
    else:
        st.pyplot(fig)


# ── 탭 3: 문서별 통계 ──
with tab3:
    st.subheader("문서별 문장 수 / 문단 수")
    if meta.empty:
        st.info("처리된 문서가 없습니다.")
    else:
        sent_df = pd.read_csv(UNIT_FILES['sentence'], encoding='utf-8-sig')
        rows = []
        for fname in meta['파일명']:
            n_sent = len(sent_df[sent_df['파일명'] == fname])
            rows.append({
                '파일명': fname,
                '문장 수': n_sent,
                '문단 수': '',   # 잠정 보류
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
