# streamlit_app.py
# Versão ajustada para fazer POST com form-data:
# data = { "acao": "consultar", "numeroProtocolo": "<protocolo>" , ...extras }
# Mantém: parsing da NOVA estrutura, concorrência, CSV/XLSX, destaque, colar HTML.

import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st

# ==============================
# Config Streamlit
# ==============================
st.set_page_config(layout="wide")
st.title("Busca de Protocolos")


# ==============================
# Configurações
# ==============================
with st.expander("⚙️ Configurações de busca", expanded=False):
    base_url = st.text_input(
        "URL de Busca",
        value="https://www.eprotocolo.pr.gov.br/spiweb/consultarProtocoloDigital.do",
        help="Endpoint que recebe POST com form-data (acao=consultar, numeroProtocolo=...).",
        disabled=True
    )
    # user_agent = st.text_input(
    #     "User-Agent",
    #     value="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    #     help="Alguns sites exigem User-Agent customizado.",
    #     # disabled
    # )
    # Extras opcionais que também irão no form-data (k=v&x=y)
    # extra_params_text = st.text_input(
    #     "Parâmetros extras (form-data opcional)",
    #     value="action=pesquisar",
    #     help="Serão enviados além de acao=consultar e numeroProtocolo. Ex.: action=pesquisar",
    #     # disabled
    # )
    extra_params_text = "action=pesquisar"
    max_workers = st.slider("Buscas Simultâneas (threads)", min_value=1, max_value=4, value=2, disabled=True)
    retries = st.slider("Tentativas por protocolo", min_value=1, max_value=3, value=2)
    backoff_base = st.number_input("Backoff (segundos) entre tentativas", min_value=0.5, value=0.7, step=0.1, disabled=True)

st.text("⚠️ O novo sistema de buscas de protocolos ainda está em fase de desenvolvimento. A contagem dos dias parados está sendo realizada manualmente pelo sistema, uma vez que o novo eProtocolo apresenta um bug nessa informação. Além disso, esta versão ainda não possui captura automática de erros, ficando sob responsabilidade do usuário a conferência dos protocolos retornados.")

# default_headers = {"User-Agent": user_agent}
default_headers = {}

# ==============================
# Entrada de protocolos
# ==============================
protocolo_input = st.text_area("Inserir protocolos (um por linha)", value="22.361.208-3")
protocol_numbers = [p.strip() for p in protocolo_input.strip().split("\n") if p.strip()]
wrong_protocol_numbers: List[str] = []


# ==============================
# Helpers
# ==============================
def _clean_text(x: str) -> str:
    return " ".join((x or "").replace("\xa0", " ").strip().split())


def process_time(x: str) -> Optional[datetime.datetime]:
    if not x:
        return None
    dt = pd.to_datetime(x, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.to_pydatetime()


def safe_to_int(dt: Optional[datetime.datetime]) -> int:
    if dt is None:
        return 0
    days_diff = (datetime.datetime.now() - dt).days
    return int(days_diff)


def color_days(value):
    try:
        v = int(value)
        if v < 1:
            return "background-color: #b7f7b2;"
        if v < 3:
            return "background-color: #d9f7be;"
        if v > 30:
            return "background-color: #ffd6e7;"
        return None
    except Exception:
        return None


def _find_value_by_label(soup: BeautifulSoup, label_text: str) -> Optional[str]:
    """
    Encontra o valor que está na COLUNA IRMÃ do <label> dentro da mesma linha (grid bootstrap).
    Funciona para blocos como:

      <div class="col-6 col-md-2 ...">
        <label title="Total Dias em Trâmite">Dias em Trâmite:</label>
      </div>
      <div class="col-6 col-md-1 ps-md-0 fw-medium">
        434
      </div>

    Estratégia:
    1) Acha o <label> cujo texto (normalizado) bate com label_text (ignora dois-pontos e case).
    2) Sobe para o DIV de coluna que contém o label (primeiro pai com classe 'col-*').
    3) Pega o PRÓXIMO irmão <div> de coluna; prioriza um que tenha 'fw-medium', senão usa o primeiro com texto útil.
    """

    def _norm(s: str) -> str:
        return " ".join((s or "").replace("\xa0", " ").strip().rstrip(":").split()).lower()

    target_norm = _norm(label_text)

    # 1) localizar o <label>
    label_node = None
    for lbl in soup.find_all("label"):
        txt = _norm(lbl.get_text(" ", strip=True))
        if txt == target_norm:
            label_node = lbl
            break
        # também aceita quando o title do label bate
        title = _norm(lbl.get("title", ""))
        if title and (title == target_norm or target_norm in title):
            label_node = lbl
            break

    if label_node is None:
        return None

    # 2) subir até o DIV de coluna (primeiro pai que tenha classe começando com 'col-')
    def _is_col_div(tag):
        if getattr(tag, "name", None) != "div":
            return False
        classes = tag.get("class", [])
        return any(cls.startswith("col-") for cls in classes)

    col_div = label_node.find_parent(_is_col_div)
    if not col_div:
        # fallback: usa a linha/row e procura .fw-medium
        row = label_node.find_parent(class_="row") or label_node.find_parent("div")
        if not row:
            return None
        cand = row.select_one(".fw-medium")
        return _clean_text(cand.get_text(" ", strip=True)) if cand else None

    # 3) procurar o próximo irmão de coluna com valor
    #    prioridade: um irmão com 'fw-medium'; senão, o primeiro irmão com texto útil
    #    ignorar nós de texto/brancos
    next_col_candidates = []
    fw_medium_candidate = None

    sib = col_div.next_sibling
    while sib is not None:
        # pular nós de texto/brancos
        if getattr(sib, "name", None) == "div":
            sib_classes = sib.get("class", [])
            if any(cls.startswith("col-") for cls in sib_classes):
                text_val = _clean_text(sib.get_text(" ", strip=True))
                if text_val:
                    next_col_candidates.append((sib, text_val))
                    if "fw-medium" in sib_classes and not fw_medium_candidate:
                        fw_medium_candidate = (sib, text_val)
                # se chegamos no próximo bloco de label, paramos
                has_label = sib.find("label") is not None
                if has_label and next_col_candidates:
                    break
        sib = sib.next_sibling

    # preferência pelo com fw-medium
    if fw_medium_candidate:
        return fw_medium_candidate[1]

    # senão, o primeiro candidato com texto
    if next_col_candidates:
        return next_col_candidates[0][1]

    # fallback: procurar dentro da mesma row pelo primeiro .fw-medium após o label
    row = col_div.find_parent(class_="row") or col_div.find_parent("div")
    if row:
        cand = row.select_one(".fw-medium")
        if cand:
            return _clean_text(cand.get_text(" ", strip=True))

    return None



def extract_hierarchy(onde_esta: str) -> Dict[str, Optional[str]]:
    if not onde_esta:
        return {"órgão_alias": None, "diretoria": None, "núcleo/grupo": None, "coordenação": None}

    parts = [p.strip() for p in onde_esta.split("-", 1)]
    orgao_alias = parts[0] if parts else None
    diretoria = nucleo = coordenacao = None

    if len(parts) > 1:
        rest = parts[1]
        subs = [s.strip() for s in rest.split("/") if s.strip()]
        if len(subs) >= 1:
            diretoria = subs[0]
        if len(subs) >= 2:
            nucleo = subs[1]
        if len(subs) >= 3:
            coordenacao = subs[2]

    return {"órgão_alias": orgao_alias, "diretoria": diretoria, "núcleo/grupo": nucleo, "coordenação": coordenacao}


def parse_eprotocolo_html(html: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(html, "lxml")

    protocolo = None
    prot_input = soup.find("input", attrs={"name": "numeroProtocolo"})
    if prot_input and prot_input.has_attr("value"):
        protocolo = _clean_text(prot_input["value"])
    if not protocolo:
        m = re.search(r"\b\d{2}\.\d{3}\.\d{3}-\d\b", soup.get_text(" ", strip=True))
        if m:
            protocolo = m.group(0)

    local_envio = _find_value_by_label(soup, "Local de Envio:")
    onde_esta = _find_value_by_label(soup, "Onde está:")
    enviado_em = _find_value_by_label(soup, "Enviado em:")
    motivo = _find_value_by_label(soup, "Motivo:")

    dias_em_tramite = _find_value_by_label(soup, "Dias em Trâmite:")

    #DIAS SOBRESTADO NÃO FUNCIONA AINDAA
    #dias_sobrestado = _find_value_by_label(soup, "Dias Sobrestado:")

    # tenta converter para datetime
    enviado_dt = process_time(enviado_em or "")

    # se conseguir converter, formata em DD/MM/YYYY e HH:MM
    enviado_em_fmt_data = enviado_dt.strftime("%d/%m/%Y") if enviado_dt else enviado_em
    enviado_em_fmt_hora = enviado_dt.strftime("%H:%M") if enviado_dt else None

    # calcula dias parado = hoje - data enviado
    if enviado_dt:
        dias_parado_calc = (datetime.datetime.now() - enviado_dt).days
    else:
        dias_parado_calc = None

    hier = extract_hierarchy(onde_esta or "")

    return {
        "protocolo": protocolo,
        "dias em tramite": _clean_text(dias_em_tramite) if dias_em_tramite else None,
        "local de envio": local_envio,
        "onde está": onde_esta,
        "data enviado": enviado_em_fmt_data,
        "hora enviado": enviado_em_fmt_hora,
        "órgão": hier.get("órgão_alias"),
        "diretoria": hier.get("diretoria"),
        "núcleo/grupo": hier.get("núcleo/grupo"),
        "coordenação": hier.get("coordenação"),
        "dias parado": dias_parado_calc,   # <-- calculado agora
        "movimentação": motivo,
        "_enviado_em_dt": enviado_dt,
    }



def _parse_kv_pairs(qs_like: str) -> Dict[str, str]:
    """
    Converte 'a=1&b=2' -> {"a":"1","b":"2"} (ignora pares sem '=' ou vazios)
    """
    out: Dict[str, str] = {}
    if not qs_like:
        return out
    for pair in qs_like.split("&"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


@st.cache_data(show_spinner=False)
def fetch_protocolo_page_post(numero_protocolo: str, base_url: str, extra_params: str, headers: Dict[str, str], retries: int, backoff: float) -> Optional[str]:
    """
    Faz POST com form-data:
      acao=consultar
      numeroProtocolo=<protocolo>
      + extras (se houver)
    """
    form = {"acao": "consultar", "numeroProtocolo": numero_protocolo}
    form.update(_parse_kv_pairs(extra_params))

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(base_url, data=form, headers=headers or {}, timeout=25)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    return None


def fetch_and_parse(numero_protocolo: str) -> Tuple[str, Optional[Dict[str, Optional[str]]]]:
    html = fetch_protocolo_page_post(
        numero_protocolo,
        base_url,
        extra_params_text,
        default_headers,
        retries,
        backoff_base,
    )

    if not html:
        return numero_protocolo, None

    data = parse_eprotocolo_html(html)
    if not data.get("protocolo"):
        data["protocolo"] = numero_protocolo
    if not data.get("dias parado"):
        data["dias parado"] = safe_to_int(data.get("_enviado_em_dt"))
    return numero_protocolo, data



def build_dataframe_from_protocolos(protocol_numbers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Optional[str]]] = []
    unique = sorted(set(protocol_numbers))
    if not unique:
        return pd.DataFrame()

    progress = st.progress(0, text="Buscando protocolos...")
    done = 0
    total = len(unique)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_and_parse, p): p for p in unique}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                _, data = fut.result()
                if data:
                    rows.append(data)
                else:
                    wrong_protocol_numbers.append(p)
            except Exception:
                wrong_protocol_numbers.append(p)
            finally:
                done += 1
                progress.progress(done / total, text=f"Processados {done}/{total}")

    progress.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    if "dias em tramite" in df.columns:
        df["dias em tramite"] = pd.to_numeric(df["dias em tramite"], errors="coerce").astype("Int64")
    if "dias parado" in df.columns:
        df["dias parado"] = pd.to_numeric(df["dias parado"], errors="coerce").fillna(0).astype(int)

    if "_enviado_em_dt" in df.columns:
        df.drop(columns=["_enviado_em_dt"], inplace=True, errors="ignore")

    df.rename(
        columns={
            "protocolo": "Protocolo",
            "dias em tramite": "Dias em trâmite",
            "local de envio": "Local de envio",
            "onde está": "Onde está",
            "data enviado": "Enviado em",
            "data enviado": "Data Enviado",
            "órgão": "Órgão",
            "diretoria": "Diretoria",
            "núcleo/grupo": "Núcleo/Grupo",
            "coordenação": "Coordenação",
            "dias parado": "Dias Parado",
            "movimentação": "Movimentação",
            "hora enviado": "Hora Enviado",
        },
        inplace=True,
    )

    if "protocolo" in df.columns:
        df.sort_values(by=["protocolo"], inplace=True, ignore_index=True)

    desired = [
        "Protocolo",
        "Dias em trâmite",
        "Local de envio",
        "Onde está",
        "Data Enviado",
        "Órgão",
        "Diretoria",
        "Núcleo/Grupo",
        "Coordenação",
        "Dias Parado",
        "Movimentação",
        "Hora Enviado"
    ]
    rest = [c for c in df.columns if c not in desired]
    df = df[[c for c in desired if c in df.columns] + rest]

    return df


def make_downloads(df: pd.DataFrame):
    from openpyxl.utils import get_column_letter

    # XLSX
    from io import BytesIO
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="protocolos", index=False)

        # pega a worksheet
        ws = writer.sheets["protocolos"]

        # ajusta largura de cada coluna de acordo com o conteúdo
        for i, col in enumerate(df.columns, 1):  # 1-indexado
            ws.column_dimensions[get_column_letter(i)].width = 50

    st.download_button("⬇️ Baixar XLSX", data=buf.getvalue(),
                       file_name="protocolos.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       use_container_width=True)
    
def is_valid_protocol(p: str) -> bool:
    return bool(PROTO_REGEX.fullmatch(p.strip()))    


def normalize_protocol(p: str) -> str:
    """Se for exatamente 9 dígitos, aplica máscara NN.NNN.NNN-N; caso contrário, retorna como veio."""
    s = re.sub(r"\D", "", p or "")
    if len(s) == 9:
        return f"{s[0:2]}.{s[2:5]}.{s[5:8]}-{s[8]}"
    return p.strip()


# ==============================
# Execução principal (POST)
# ==============================

PROTO_REGEX = re.compile(r"^\d{2}\.\d{3}\.\d{3}-\d$")

def normalize_protocol(p: str) -> str:
    """Se for exatamente 9 dígitos, aplica máscara NN.NNN.NNN-N; caso contrário, retorna como veio."""
    s = re.sub(r"\D", "", p or "")
    if len(s) == 9:
        return f"{s[0:2]}.{s[2:5]}.{s[5:8]}-{s[8]}"
    return p.strip()

def is_valid_protocol(p: str) -> bool:
    return bool(PROTO_REGEX.fullmatch(p.strip()))


if st.button("Buscar protocolos (POST)", type="primary", disabled=(len(protocol_numbers) == 0)):
    start = datetime.datetime.now()

    # normaliza todos os protocolos de entrada
    input_raw = [p for p in protocol_numbers if p.strip()]
    normalized = [normalize_protocol(p) for p in input_raw]

    # mapeia original → normalizado
    norm_map = {orig: norm for orig, norm in zip(input_raw, normalized)}

    # identifica protocolos normalizados
    normalized_pairs = [(orig, norm) for orig, norm in norm_map.items()
                        if orig != norm and is_valid_protocol(norm)]

    # identifica protocolos realmente inválidos (nem com máscara bate regex)
    invalid_reals = sorted({orig for orig, norm in norm_map.items() if not is_valid_protocol(norm)})

    # apenas válidos seguem para busca
    to_fetch = [p for p in normalized if is_valid_protocol(p)]
    df = build_dataframe_from_protocolos(to_fetch)

    elapsed = (datetime.datetime.now() - start).seconds
    st.write(f"Tempo decorrido: {elapsed} s")

    if df.empty:
        st.warning("Nenhuma informação encontrada para os protocolos informados.")
    else:
        column_order = [
            "Protocolo",
            "Dias em trâmite",
            "Local de envio",
            "Onde está",
            "Data Enviado",
            "Órgão",
            "Diretoria",
            "Núcleo/Grupo",
            "Coordenação",
            "Dias Parado",
            "Movimentação",
            "Hora Enviado"
        ]
        show_cols = [c for c in column_order if c in df.columns] + [c for c in df.columns if c not in column_order]

        styled = df.style
        if "Dias Parado" in df.columns:
            styled = styled.applymap(color_days, subset=["Dias Parado"])

        st.dataframe(styled, use_container_width=True, hide_index=True, column_order=show_cols)

        # identifica protocolos que não retornaram no DF
        ok = set(df["Protocolo"].dropna().astype(str).str.strip().tolist()) if "Protocolo" in df.columns else set()
        missing = sorted([p for p in to_fetch if p not in ok])

        # if invalid_reals or missing:
        #     st.write("### Protocolos com erro")
        #     if invalid_reals:
        #         st.write(f"• **Formato inválido**: {invalid_reals}")
        #     if missing:
        #         st.write(f"• **Não encontrados/sem retorno**: {missing}")
        # else:
        #     st.write("Protocolos com erro - nenhum")

        make_downloads(df)
