"""
Pipeline local de classificação de licitações via OpenAI.
Execução: python classificacao_licitacoes.py

Configuração via variáveis de ambiente:
  OPENAI_API_KEY=sk-...
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import psycopg2
import requests
from groq import Groq
from psycopg2.extras import execute_values
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CONFIGURAÇÕES

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "srv1236151.hstgr.cloud"),
    "port": int(os.getenv("DB_PORT", 5433)),
    "database": os.getenv("DB_NAME", "aula"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "curso_python"),
}

PNCP_BASE_URL = "https://pncp.gov.br/api/consulta/v1/contratacoes/publicacao"
UF_FILTRO = "RO"
CODIGO_MODALIDADE_CONTRATACAO = 1
TAMANHO_PAG = 50
DIAS_JANELA = 365
PNCP_TIMEOUT = 90
PNCP_MAX_TENTATIVAS = 3

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_MODEL = "llama-3.3-70b-versatile"
BATCH_SIZE = 10

CATEGORIAS = [
    "Saúde", "Educação", "Infraestrutura e Obras", "Tecnologia da Informação",
    "Alimentação e Nutrição", "Segurança Pública", "Meio Ambiente e Saneamento",
    "Transporte e Logística", "Administrativo e Material de Escritório",
    "Serviços Gerais", "Outros",
]


# HELPERS

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, client_encoding="utf8")


def criar_tabelas():
    ddl_licitacoes = """
        CREATE TABLE IF NOT EXISTS licitacoes_pncp (
            id                      BIGSERIAL PRIMARY KEY,
            numero_controle_pncp    TEXT UNIQUE,
            objeto_compra           TEXT,
            modalidade_nome         TEXT,
            orgao_nome              TEXT,
            orgao_cnpj              TEXT,
            uf                      TEXT,
            valor_total_estimado    NUMERIC(18,2),
            data_publicacao         DATE,
            data_abertura_proposta  TIMESTAMP,
            situacao                TEXT,
            link_sistema_origem     TEXT,
            json_original           JSONB,
            inserido_em             TIMESTAMP DEFAULT NOW()
        );
    """
    ddl_classificacoes = """
        CREATE TABLE IF NOT EXISTS licitacoes_classificadas (
            id                   BIGSERIAL PRIMARY KEY,
            numero_controle_pncp TEXT,
            objeto_compra        TEXT,
            orgao_nome           TEXT,
            uf                   TEXT,
            categoria            TEXT,
            confianca            TEXT,
            objeto_vago          BOOLEAN,
            justificativa_vago   TEXT,
            resumo               TEXT,
            tokens_usados        INTEGER,
            modelo_usado         TEXT,
            provider_llm         TEXT,
            data_publicacao      DATE,
            classificado_em      TIMESTAMP DEFAULT NOW()
        );
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl_licitacoes)
            cur.execute(ddl_classificacoes)
        conn.commit()
    logger.info("Tabelas verificadas/criadas.")


# TASK 1 - EXTRAÇÃO

def extrair_licitacoes():
    hoje = datetime.now()
    data_fim = hoje.strftime("%Y%m%d")
    data_ini = (hoje - timedelta(days=DIAS_JANELA)).strftime("%Y%m%d")

    logger.info(f"Buscando licitações PNCP | UF={UF_FILTRO} | {data_ini} -> {data_fim}")

    todas, pagina = [], 1

    while True:
        params = {
            "dataInicial": data_ini,
            "dataFinal": data_fim,
            "codigoModalidadeContratacao": CODIGO_MODALIDADE_CONTRATACAO,
            "uf": UF_FILTRO,
            "pagina": pagina,
            "tamanhoPagina": TAMANHO_PAG,
        }
        resp = None
        for tentativa in range(1, PNCP_MAX_TENTATIVAS + 1):
            try:
                resp = requests.get(PNCP_BASE_URL, params=params, timeout=PNCP_TIMEOUT)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Tentativa {tentativa}/{PNCP_MAX_TENTATIVAS} falhou na página {pagina}: {e}"
                )
                if tentativa == PNCP_MAX_TENTATIVAS:
                    logger.error(f"Erro definitivo na página {pagina}: {e}")
                    return todas
                time.sleep(2)

        if resp.status_code == 204:
            logger.info(f"Página {pagina} sem resultados para os filtros informados.")
            break

        try:
            payload = resp.json()
        except RequestsJSONDecodeError:
            content_type = resp.headers.get("Content-Type", "desconhecido")
            corpo = (resp.text or "").strip().replace("\n", " ")[:300]
            logger.error(
                f"Resposta inválida na página {pagina} | status={resp.status_code} "
                f"| content-type={content_type} | corpo='{corpo}'"
            )
            return todas

        registros = payload.get("data", [])
        total = payload.get("totalRegistros", 0)

        if pagina == 1:
            logger.info(f"Total de licitações encontradas: {total}")
        if not registros:
            break

        todas.extend(registros)
        logger.info(f"Página {pagina} - {len(registros)} registros | acumulado: {len(todas)}")

        total_paginas = -(-total // TAMANHO_PAG)
        if pagina >= total_paginas or pagina >= 20:
            break

        pagina += 1
        time.sleep(0.3)

    logger.info(f"Extração concluída: {len(todas)} licitações")
    return todas


# TASK 2 - ARMAZENAMENTO

def salvar_postgres(licitacoes):
    if not licitacoes:
        logger.warning("Nenhuma licitação para salvar.")
        return [], 0

    criar_tabelas()

    def parse_data(val):
        if not val:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(val[:19], fmt)
            except ValueError:
                continue
        return None

    registros = []
    for l in licitacoes:
        orgao = l.get("orgaoEntidade", {}) or {}
        unidade = l.get("unidadeOrgao", {}) or {}
        registros.append((
            l.get("numeroControlePNCP"),
            l.get("objetoCompra"),
            l.get("modalidadeNome"),
            orgao.get("razaoSocial"),
            orgao.get("cnpj"),
            unidade.get("ufSigla", UF_FILTRO),
            l.get("valorTotalEstimado"),
            parse_data(l.get("dataPublicacaoPncp")),
            parse_data(l.get("dataAberturaProposta")),
            l.get("situacaoCompraNome"),
            l.get("linkSistemaOrigem"),
            json.dumps(l, ensure_ascii=False),
        ))

    sql = """
        INSERT INTO licitacoes_pncp (
            numero_controle_pncp, objeto_compra, modalidade_nome,
            orgao_nome, orgao_cnpj, uf, valor_total_estimado,
            data_publicacao, data_abertura_proposta, situacao,
            link_sistema_origem, json_original
        ) VALUES %s
        ON CONFLICT (numero_controle_pncp) DO NOTHING
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, registros, page_size=500)
            inseridos = cur.rowcount
        conn.commit()

    logger.info(f"{len(registros)} processadas | {inseridos} novas inseridas.")

    objetos = [
        {
            "numero_controle_pncp": l.get("numeroControlePNCP"),
            "objeto_compra": l.get("objetoCompra", ""),
            "orgao_nome": (l.get("orgaoEntidade") or {}).get("razaoSocial", ""),
            "uf": (l.get("unidadeOrgao") or {}).get("ufSigla", UF_FILTRO),
            "data_publicacao": l.get("dataPublicacaoPncp", "")[:10] if l.get("dataPublicacaoPncp") else None,
        }
        for l in licitacoes if l.get("objetoCompra")
    ]
    return objetos, inseridos


# TASK 3 - CLASSIFICAÇÃO COM OPENAI

def _chamar_openai(prompt_sistema, prompt_usuario):
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": prompt_usuario},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},  # suportado pelo Groq
    )
    return {
        "conteudo": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "modelo": OPENAI_MODEL,
    }


def _montar_prompt(objeto, categorias):
    cats_formatadas = "\n".join(f"  - {c}" for c in categorias)
    prompt_sistema = f"""Você é um especialista em transparência pública e licitações governamentais brasileiras.

Sua tarefa é analisar o objeto de uma licitação pública e retornar um JSON com:

{{
  "categoria":         "uma das categorias listadas abaixo",
  "confianca":         "ALTA | MÉDIA | BAIXA",
  "objeto_vago":       true | false,
  "justificativa_vago": "por que é vago (se objeto_vago=true) ou string vazia",
  "resumo":            "1 frase resumindo o objeto em linguagem simples"
}}

CATEGORIAS DISPONÍVEIS:
{cats_formatadas}

CRITÉRIOS PARA objeto_vago=true:
  - Objeto genérico demais (ex: "aquisição de materiais diversos")
  - Ausência de especificação do que será adquirido/contratado
  - Termos vagos como "conforme termo de referência" sem mais detalhes
  - Descrição com menos de 5 palavras informativas

REGRAS:
  - Responda APENAS com o JSON, sem texto adicional
  - Se não souber a categoria, use "Outros"
  - Confiança BAIXA quando o objeto for muito vago para classificar
  - Resumo deve ter no máximo 15 palavras"""

    prompt_usuario = f'Analise este objeto de licitação:\n\n"{objeto}"'
    return prompt_sistema, prompt_usuario


def classificar_com_llm(objetos):
    if not objetos:
        logger.warning("Nenhum objeto para classificar.")
        return []

    logger.info(f"Classificando {len(objetos)} licitações | Provider: groq | Modelo: {OPENAI_MODEL}")

    classificacoes, erros, tokens_total = [], 0, 0

    for i, item in enumerate(objetos):
        objeto = item.get("objeto_compra", "").strip()
        if not objeto or len(objeto) < 5:
            continue

        try:
            prompt_sis, prompt_usr = _montar_prompt(objeto, CATEGORIAS)
            resultado = _chamar_openai(prompt_sis, prompt_usr)

            match = re.search(r"\{.*\}", resultado["conteudo"].strip(), re.DOTALL)
            if not match:
                raise ValueError(f"Nenhum JSON encontrado: {resultado['conteudo'][:100]}")

            dados = json.loads(match.group())
            categoria = dados.get("categoria", "Outros")
            if categoria not in CATEGORIAS:
                categoria = "Outros"

            confianca = dados.get("confianca", "BAIXA").upper().replace("MEDIA", "MÉDIA")
            if confianca not in ("ALTA", "MÉDIA", "BAIXA"):
                confianca = "BAIXA"

            classificacoes.append({
                **item,
                "categoria": categoria,
                "confianca": confianca,
                "objeto_vago": bool(dados.get("objeto_vago", False)),
                "justificativa_vago": dados.get("justificativa_vago", ""),
                "resumo": dados.get("resumo", ""),
                "tokens_usados": resultado["tokens"],
                "modelo_usado": resultado["modelo"],
                "provider_llm": "groq",
            })
            tokens_total += resultado["tokens"]

            if (i + 1) % 10 == 0:
                vagos_ate_agora = sum(1 for c in classificacoes if c["objeto_vago"])
                logger.info(f"  Progresso: {i+1}/{len(objetos)} | Tokens: {tokens_total} | Vagos: {vagos_ate_agora}")

        except Exception as e:
            erros += 1
            logger.error(f"[{i+1}] Erro ao classificar '{objeto[:60]}': {e}")
            classificacoes.append({
                **item,
                "categoria": "Outros",
                "confianca": "BAIXA",
                "objeto_vago": False,
                "justificativa_vago": "",
                "resumo": f"Erro na classificação: {str(e)[:100]}",
                "tokens_usados": 0,
                "modelo_usado": OPENAI_MODEL,
                "provider_llm": "groq",
            })

        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(1)

    vagos = sum(1 for c in classificacoes if c.get("objeto_vago"))
    logger.info(
        f"Classificação concluída: {len(classificacoes)} | "
        f"Vagos: {vagos} ({vagos/len(classificacoes)*100:.1f}%) | "
        f"Erros: {erros} | Tokens: {tokens_total}"
    )
    return classificacoes


# TASK 4 - SALVAR CLASSIFICAÇÕES

def salvar_classificacoes(classificacoes):
    if not classificacoes:
        logger.info("Nenhuma classificação para salvar.")
        return 0

   # sql_truncate = "TRUNCATE TABLE licitacoes_classificadas RESTART IDENTITY;"
    sql_insert = """
        INSERT INTO licitacoes_classificadas (
            numero_controle_pncp, objeto_compra, orgao_nome, uf,
            categoria, confianca, objeto_vago, justificativa_vago,
            resumo, tokens_usados, modelo_usado, provider_llm, data_publicacao
        ) VALUES %s
    """
    registros = [
        (
            c.get("numero_controle_pncp"),
            c.get("objeto_compra"),
            c.get("orgao_nome"),
            c.get("uf"),
            c.get("categoria"),
            c.get("confianca"),
            c.get("objeto_vago"),
            c.get("justificativa_vago"),
            c.get("resumo"),
            c.get("tokens_usados", 0),
            c.get("modelo_usado"),
            c.get("provider_llm"),
            c.get("data_publicacao"),
        )
        for c in classificacoes
    ]

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # cur.execute(sql_truncate)
            execute_values(cur, sql_insert, registros)
        conn.commit()

    vagos = sum(1 for c in classificacoes if c.get("objeto_vago"))
    logger.info(f"{len(registros)} classificações salvas | {vagos} objetos vagos.")
    return len(registros)


# TASK 5 - RELATÓRIO HTML

def gerar_relatorio(classificacoes):
    if not classificacoes:
        logger.info("Sem dados para relatório.")
        return

    total = len(classificacoes)
    vagos = [c for c in classificacoes if c.get("objeto_vago")]
    por_categoria = {}
    for c in classificacoes:
        cat = c.get("categoria", "Outros")
        por_categoria[cat] = por_categoria.get(cat, 0) + 1
    cats_ordenadas = sorted(por_categoria.items(), key=lambda x: x[1], reverse=True)
    tokens_total = sum(c.get("tokens_usados", 0) for c in classificacoes)

    linhas_vagos = ""
    for v in sorted(vagos, key=lambda x: x.get("orgao_nome", ""))[:50]:
        objeto = (v.get("objeto_compra") or "")[:120]
        linhas_vagos += f"""
        <tr>
          <td>{v.get('orgao_nome', '')[:50]}</td>
          <td>{objeto}{"..." if len(v.get("objeto_compra", "")) > 120 else ""}</td>
          <td>{v.get('categoria', '')}</td>
          <td>{v.get('justificativa_vago', '')[:100]}</td>
        </tr>"""

    max_cat = cats_ordenadas[0][1] if cats_ordenadas else 1
    barras_html = ""
    for cat, qtd in cats_ordenadas:
        pct = round(qtd / total * 100, 1)
        largura = round(qtd / max_cat * 100)
        barras_html += f"""
        <div class="bar-row">
          <span class="bar-label">{cat}</span>
          <div class="bar-wrap"><div class="bar-fill" style="width:{largura}%"></div></div>
          <span class="bar-count">{qtd} ({pct}%)</span>
        </div>"""

    modelo_usado = classificacoes[0].get("modelo_usado", "?")
    data_exec = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Relatório de Licitações - {data_exec}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background:#f5f7fa; color:#333; }}
  .header {{ background:#0D2B55; color:white; padding:24px 32px; border-radius:8px; margin-bottom:24px; }}
  .header h1 {{ margin:0; font-size:22px; }}
  .header p  {{ margin:4px 0 0; opacity:.75; font-size:13px; }}
  .cards {{ display:flex; gap:16px; margin-bottom:24px; flex-wrap:wrap; }}
  .card {{ background:white; border-radius:8px; padding:20px 24px; flex:1; min-width:140px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .card .num {{ font-size:32px; font-weight:700; color:#0D2B55; }}
  .card .lbl {{ font-size:12px; color:#666; margin-top:4px; }}
  .card.alert .num {{ color:#C62828; }}
  .card.info  .num {{ color:#00838F; }}
  .section {{ background:white; border-radius:8px; padding:24px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,.08); }}
  .section h2 {{ margin-top:0; font-size:16px; color:#0D2B55; border-bottom:2px solid #E3F2FD; padding-bottom:10px; }}
  .bar-row {{ display:flex; align-items:center; gap:12px; margin-bottom:8px; }}
  .bar-label {{ width:220px; font-size:13px; text-align:right; color:#555; }}
  .bar-wrap  {{ flex:1; background:#E8ECEF; border-radius:4px; height:18px; }}
  .bar-fill  {{ background:#1565C0; border-radius:4px; height:100%; }}
  .bar-count {{ width:90px; font-size:12px; color:#888; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ background:#0D2B55; color:white; padding:10px 12px; text-align:left; }}
  td {{ padding:9px 12px; border-bottom:1px solid #EEE; vertical-align:top; }}
  tr:hover td {{ background:#F5F9FF; }}
  .footer {{ text-align:center; font-size:11px; color:#AAA; margin-top:24px; }}
</style>
</head>
<body>
<div class="header">
  <h1>Relatório de Classificação de Licitações - Ceará</h1>
  <p>Gerado localmente | {data_exec} | Modelo: {modelo_usado}</p>
</div>
<div class="cards">
  <div class="card"><div class="num">{total}</div><div class="lbl">Licitações analisadas</div></div>
  <div class="card alert"><div class="num">{len(vagos)}</div><div class="lbl">Objetos vagos detectados</div></div>
  <div class="card info"><div class="num">{len(cats_ordenadas)}</div><div class="lbl">Categorias identificadas</div></div>
  <div class="card"><div class="num">{tokens_total:,}</div><div class="lbl">Tokens LLM consumidos</div></div>
</div>
<div class="section">
  <h2>Distribuição por Categoria Temática</h2>
  {barras_html}
</div>
<div class="section">
  <h2>Objetos com Descrição Vaga ({len(vagos)} licitações)</h2>
  <table>
    <thead><tr><th>Órgão</th><th>Objeto original</th><th>Categoria</th><th>Por que é vago</th></tr></thead>
    <tbody>{linhas_vagos}</tbody>
  </table>
</div>
<div class="footer">Pipeline classificacao_licitacoes - Execução local</div>
</body>
</html>"""

    caminho = Path("relatorio_licitacoes.html")
    caminho.write_text(html, encoding="utf-8")
    logger.info(f"Relatório salvo em {caminho.resolve()} ({len(html)} bytes)")
    return str(caminho)


# EXECUÇÃO LOCAL

if __name__ == "__main__":
    licitacoes = extrair_licitacoes()
    objetos, _ = salvar_postgres(licitacoes)
    classificacoes = classificar_com_llm(objetos)
    salvar_classificacoes(classificacoes)
    gerar_relatorio(classificacoes)