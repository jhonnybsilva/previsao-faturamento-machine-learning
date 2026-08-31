"""
=============================================================================
PROJETO: ANÁLISE DE VENDAS + PREVISÃO DE FATURAMENTO COM MACHINE LEARNING
MÓDULO: EDA, ENGENHARIA DE FEATURES, MODELAGEM PREDITIVA E FORECAST 3 MESES
AUTOR: Jhonny Brasiliano da Silva
=============================================================================
"""

import sys
import os
import sqlite3
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuração visual dos gráficos
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.dpi'] = 120

def executar_projeto():
    print("=" * 75)
    print("🚀 INICIANDO ANÁLISE DE VENDAS & MACHINE LEARNING DE FATURAMENTO")
    print("👤 Desenvolvedor: Jhonny Brasiliano da Silva")
    print("=" * 75)

    # 1. Conexão com o Banco SQLite Local
    db_path = "vendas_empresa.db"
    if not os.path.exists(db_path):
        print(f"[ERRO] Banco '{db_path}' não encontrado. Execute 'gerar_banco_dados.py' primeiro.")
        return

    conn = sqlite3.connect(db_path)
    
    # 2. Extração de Dados via SQL
    query_vendas = """
        SELECT 
            id_venda,
            data_venda,
            ano,
            mes,
            ano_mes,
            regiao,
            canal,
            metodo_pagamento,
            status,
            valor_bruto,
            desconto,
            valor_liquido,
            lucro_liquido
        FROM vendas
        WHERE status = 'Concluido'
        ORDER BY data_venda ASC
    """
    df_vendas = pd.read_sql_query(query_vendas, conn)
    
    query_itens = """
        SELECT 
            iv.id_item,
            iv.id_venda,
            iv.nome_produto,
            iv.categoria,
            iv.quantidade,
            iv.preco_unitario,
            iv.subtotal,
            iv.lucro_bruto,
            v.data_venda,
            v.ano_mes
        FROM itens_venda iv
        JOIN vendas v ON iv.id_venda = v.id_venda
        WHERE v.status = 'Concluido'
    """
    df_itens = pd.read_sql_query(query_itens, conn)
    conn.close()

    print(f"\n[1/5] Dados carregados do banco SQLite:")
    print(f"  • Total de Vendas Concluídas: {len(df_vendas):,} pedidos")
    print(f"  • Total de Itens Vendidos: {df_itens['quantidade'].sum():,} unidades")
    print(f"  • Faturamento Total Histórico: R$ {df_vendas['valor_liquido'].sum():,.2f}")
    print(f"  • Lucro Líquido Total: R$ {df_vendas['lucro_liquido'].sum():,.2f}")
    print(f"  • Ticket Médio por Pedido: R$ {df_vendas['valor_liquido'].mean():,.2f}")

    # =========================================================================
    # 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
    # =========================================================================
    print("\n[2/5] Realizando Análise Exploratória e Insights de Negócio...")
    
    # Faturamento por Categoria
    df_cat = df_itens.groupby('categoria').agg(
        faturamento=('subtotal', 'sum'),
        lucro=('lucro_bruto', 'sum'),
        itens_vendidos=('quantidade', 'sum')
    ).reset_index().sort_values('faturamento', ascending=False)
    
    # Faturamento por Região
    df_regiao = df_vendas.groupby('regiao')['valor_liquido'].sum().reset_index().sort_values('valor_liquido', ascending=False)
    
    # Faturamento por Canal
    df_canal = df_vendas.groupby('canal')['valor_liquido'].sum().reset_index().sort_values('valor_liquido', ascending=False)

    # Gráfico 1: Categorias e Regiões
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.barplot(data=df_cat, x='faturamento', y='categoria', palette='Blues_r', ax=axes[0])
    axes[0].set_title("Faturamento por Categoria de Produto (R$)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Faturamento Total Acumulado (R$)")
    axes[0].set_ylabel("")
    for p in axes[0].patches:
        width = p.get_width()
        axes[0].annotate(f'R$ {width/1e6:.2f}M', (width, p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9, fontweight='bold')

    sns.barplot(data=df_regiao, x='valor_liquido', y='regiao', palette='Purples_r', ax=axes[1])
    axes[1].set_title("Faturamento por Região (R$)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Faturamento Total (R$)")
    axes[1].set_ylabel("")
    for p in axes[1].patches:
        width = p.get_width()
        axes[1].annotate(f'R$ {width/1e6:.2f}M', (width, p.get_y() + p.get_height() / 2.),
                         ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("grafico_eda_categorias.png", dpi=150)
    plt.close()
    print("  [OK] Gráfico 'grafico_eda_categorias.png' salvo.")

    # =========================================================================
    # 4. ENGENHARIA DE FEATURES TEMPORAIS (SÉRIE TEMPORAL MENSAL)
    # =========================================================================
    print("\n[3/5] Estruturando Série Temporal e Engenharia de Features para Machine Learning...")
    
    df_mensal = df_vendas.groupby(['ano', 'mes', 'ano_mes']).agg(
        faturamento=('valor_liquido', 'sum'),
        lucro=('lucro_liquido', 'sum'),
        qtd_pedidos=('id_venda', 'count'),
        ticket_medio=('valor_liquido', 'mean')
    ).reset_index()

    # Ordenar cronologicamente
    df_mensal = df_mensal.sort_values(['ano', 'mes']).reset_index(drop=True)
    df_mensal['indice_tempo'] = np.arange(len(df_mensal)) # Variável de tendência linear

    # Criação de Lags (Valores de meses anteriores)
    df_mensal['lag_1'] = df_mensal['faturamento'].shift(1)  # Mês anterior
    df_mensal['lag_2'] = df_mensal['faturamento'].shift(2)  # 2 meses atrás
    df_mensal['lag_3'] = df_mensal['faturamento'].shift(3)  # 3 meses atrás
    df_mensal['lag_12'] = df_mensal['faturamento'].shift(12) # Mesmo mês do ano anterior (sazonalidade anual)

    # Médias Móveis
    df_mensal['media_movel_3m'] = df_mensal['faturamento'].shift(1).rolling(window=3).mean()
    df_mensal['media_movel_6m'] = df_mensal['faturamento'].shift(1).rolling(window=6).mean()

    # Sazonalidade Trigonométrica (Ciclo Anual de 12 meses)
    df_mensal['sin_mes'] = np.sin(2 * np.pi * df_mensal['mes'] / 12)
    df_mensal['cos_mes'] = np.cos(2 * np.pi * df_mensal['mes'] / 12)

    # Variáveis binárias para meses de alto impacto comercial
    df_mensal['is_black_friday'] = (df_mensal['mes'] == 11).astype(int)
    df_mensal['is_natal'] = (df_mensal['mes'] == 12).astype(int)
    df_mensal['is_dia_das_maes'] = (df_mensal['mes'] == 5).astype(int)
    df_mensal['trimestre'] = ((df_mensal['mes'] - 1) // 3) + 1

    # Remove linhas iniciais com NaN devido aos lags (primeiros 12 meses para ter lag_12 limpo)
    df_modelo = df_mensal.dropna().copy().reset_index(drop=True)
    
    features = [
        'indice_tempo', 'mes', 'trimestre', 'sin_mes', 'cos_mes',
        'is_black_friday', 'is_natal', 'is_dia_das_maes',
        'lag_1', 'lag_2', 'lag_3', 'lag_12',
        'media_movel_3m', 'media_movel_6m'
    ]
    target = 'faturamento'

    X = df_modelo[features]
    y = df_modelo[target]

    print(f"  • Total de meses históricos analisados: {len(df_mensal)} meses")
    print(f"  • Período do dataset com histórico de features: {df_modelo['ano_mes'].iloc[0]} a {df_modelo['ano_mes'].iloc[-1]}")
    print(f"  • Total de features preditivas criadas: {len(features)}")

    # =========================================================================
    # 5. TREINAMENTO E COMPARAÇÃO DE MODELOS DE MACHINE LEARNING
    # =========================================================================
    print("\n[4/5] Treinando Modelos de Machine Learning e Avaliando Performance...")

    # Divisão Treino e Teste Temporal (últimos 6 meses para teste / validação)
    n_teste = 6
    X_train, X_test = X.iloc[:-n_teste], X.iloc[-n_teste:]
    y_train, y_test = y.iloc[:-n_teste], y.iloc[-n_teste:]
    meses_teste = df_modelo['ano_mes'].iloc[-n_teste:].values

    modelos = {
        "Regressão Linear Múltipla": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    }

    resultados = []
    previsoes_teste = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        previsoes_teste[nome] = y_pred
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        resultados.append({
            "Modelo": nome,
            "R² Score": r2,
            "MAE (Erro Médio Absoluto)": mae,
            "RMSE": rmse,
            "MAPE (%)": mape
        })
        print(f"  • {nome:30} -> R²: {r2:.4f} | MAE: R$ {mae:,.2f} | MAPE: {mape:.2f}%")

    df_resultados = pd.DataFrame(resultados).sort_values("R² Score", ascending=False)
    
    # Seleciona o melhor modelo (Gradient Boosting ou Random Forest)
    melhor_nome = df_resultados.iloc[0]["Modelo"]
    melhor_modelo = modelos[melhor_nome]
    print(f"\n [CAMPEÃO] Melhor Modelo Selecionado: '{melhor_nome}' (R² = {df_resultados.iloc[0]['R² Score']:.4f})")

    # Retreinando o melhor modelo com 100% dos dados históricos disponíveis
    melhor_modelo.fit(X, y)

    # =========================================================================
    # 6. PREVISÃO DE FATURAMENTO PARA OS PRÓXIMOS 3 MESES (Set, Out e Nov/2026)
    # =========================================================================
    print("\n[5/5] Gerando Previsão Preditiva para os Próximos 3 Meses...")

    # Data de corte: Agosto/2026 -> Prever Setembro/2026, Outubro/2026 e Novembro/2026
    historico_faturamento = list(df_mensal['faturamento'].values)
    proximos_meses_info = [
        {"ano": 2026, "mes": 9, "nome_mes": "Setembro/2026"},
        {"ano": 2026, "mes": 10, "nome_mes": "Outubro/2026"},
        {"ano": 2026, "mes": 11, "nome_mes": "Novembro/2026"} # Black Friday!
    ]

    previsoes_futuras = []
    
    # Pipeline de previsão recursiva com atualização de lags dinâmicos
    for i, prox in enumerate(proximos_meses_info):
        idx_tempo = len(df_mensal) + i
        mes = prox["mes"]
        ano = prox["ano"]
        
        # Cálculo das features para o mês futuro
        lag_1 = historico_faturamento[-1]
        lag_2 = historico_faturamento[-2]
        lag_3 = historico_faturamento[-3]
        lag_12 = historico_faturamento[-12]
        
        mm_3m = np.mean(historico_faturamento[-3:])
        mm_6m = np.mean(historico_faturamento[-6:])
        
        sin_m = np.sin(2 * np.pi * mes / 12)
        cos_m = np.cos(2 * np.pi * mes / 12)
        
        is_bf = 1 if mes == 11 else 0
        is_nat = 1 if mes == 12 else 0
        is_mae = 1 if mes == 5 else 0
        trim = ((mes - 1) // 3) + 1
        
        linha_futura = pd.DataFrame([{
            'indice_tempo': idx_tempo,
            'mes': mes,
            'trimestre': trim,
            'sin_mes': sin_m,
            'cos_mes': cos_m,
            'is_black_friday': is_bf,
            'is_natal': is_nat,
            'is_dia_das_maes': is_mae,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'lag_12': lag_12,
            'media_movel_3m': mm_3m,
            'media_movel_6m': mm_6m
        }])
        
        pred_valor = float(melhor_modelo.predict(linha_futura)[0])
        
        # Intervalo de Confiança (Margem de erro baseada no MAPE do modelo)
        margem_erro = pred_valor * (df_resultados.iloc[0]["MAPE (%)"] / 100.0)
        cenario_pessimista = round(pred_valor - margem_erro, 2)
        cenario_otimista = round(pred_valor + margem_erro, 2)
        
        previsoes_futuras.append({
            "periodo": prox["nome_mes"],
            "mes_num": mes,
            "ano": ano,
            "faturamento_previsto": round(pred_valor, 2),
            "cenario_pessimista": cenario_pessimista,
            "cenario_otimista": cenario_otimista,
            "margem_erro": round(margem_erro, 2)
        })
        
        # Adiciona previsão à lista para alimentar os próximos lags
        historico_faturamento.append(pred_valor)

    df_previsoes = pd.DataFrame(previsoes_futuras)
    faturamento_total_3m = df_previsoes['faturamento_previsto'].sum()

    print("\n" + "=" * 75)
    print("📊 RESULTADO DA PREVISÃO DE FATURAMENTO (PRÓXIMOS 3 MESES):")
    print("=" * 75)
    for p in previsoes_futuras:
        print(f"  📅 {p['periodo']:16} -> Previsão: R$ {p['faturamento_previsto']:>12,.2f}  (Min: R$ {p['cenario_pessimista']:,.2f} | Max: R$ {p['cenario_otimista']:,.2f})")
    print("-" * 75)
    print(f"💰 FATURAMENTO ESTIMADO TOTAL NO TRIMESTRE: R$ {faturamento_total_3m:,.2f}")
    print("=" * 75)

    # =========================================================================
    # 7. GERAÇÃO DE GRÁFICO FINAL (HISTÓRICO + PREVISÃO FUTURA)
    # =========================================================================
    plt.figure(figsize=(14, 6))
    
    datas_historicas = df_mensal['ano_mes'].values
    faturamento_historico = df_mensal['faturamento'].values
    
    plt.plot(datas_historicas, faturamento_historico, marker='o', color='#3b82f6', linewidth=2.5, label='Faturamento Real Histórico')
    
    # Plot da Previsão
    datas_futuras = [p["periodo"][:3] + "/" + str(p["ano"])[2:] for p in previsoes_futuras]
    datas_futuras_completas = np.concatenate([[datas_historicas[-1]], datas_futuras])
    valores_futuros_completos = np.concatenate([[faturamento_historico[-1]], df_previsoes['faturamento_previsto'].values])
    
    plt.plot(datas_futuras_completas, valores_futuros_completos, marker='s', color='#10b981', linestyle='--', linewidth=3, label=f'Previsão IA ({melhor_nome})')
    
    # Área sombreada de intervalo de confiança
    min_vals = np.concatenate([[faturamento_historico[-1]], df_previsoes['cenario_pessimista'].values])
    max_vals = np.concatenate([[faturamento_historico[-1]], df_previsoes['cenario_otimista'].values])
    plt.fill_between(datas_futuras_completas, min_vals, max_vals, color='#10b981', alpha=0.2, label='Intervalo de Confiança (Cenário Min/Max)')
    
    # Destacar os 3 pontos futuros
    for i, p in enumerate(previsoes_futuras):
        plt.annotate(f"R$ {p['faturamento_previsto']/1e3:.1f}k", 
                     (datas_futuras[i], p['faturamento_previsto']),
                     textcoords="offset points", xytext=(0, 12), ha='center',
                     fontweight='bold', color='#065f46', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', fc='#d1fae5', ec='#10b981', lw=1.5))

    plt.title("Histórico de Vendas e Previsão de Faturamento para os Próximos 3 Meses", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Mês / Ano", fontsize=11, fontweight='bold')
    plt.ylabel("Faturamento Líquido (R$)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig("grafico_previsao_faturamento.png", dpi=150)
    plt.close()
    print("\n[OK] Gráfico executivo 'grafico_previsao_faturamento.png' gerado com sucesso!")

    # 8. Exportar Resumo dos Resultados para JSON (para alimentar dashboard / relatórios)
    dados_resumo = {
        "metricas_gerais": {
            "total_pedidos": int(len(df_vendas)),
            "faturamento_acumulado": float(df_vendas['valor_liquido'].sum()),
            "lucro_acumulado": float(df_vendas['lucro_liquido'].sum()),
            "ticket_medio": float(df_vendas['valor_liquido'].mean()),
            "meses_analisados": int(len(df_mensal))
        },
        "melhor_modelo": {
            "nome": melhor_nome,
            "r2_score": float(df_resultados.iloc[0]["R² Score"]),
            "mae": float(df_resultados.iloc[0]["MAE (Erro Médio Absoluto)"]),
            "mape": float(df_resultados.iloc[0]["MAPE (%)"])
        },
        "comparacao_modelos": df_resultados.to_dict(orient="records"),
        "previsoes_3_meses": previsoes_futuras,
        "faturamento_total_previsto_trimestre": float(faturamento_total_3m)
    }

    with open("resultado_previsao.json", "w", encoding="utf-8") as f:
        json.dump(dados_resumo, f, indent=4, ensure_ascii=False)
        
    print("[OK] Arquivo 'resultado_previsao.json' salvo com todas as métricas estruturadas.")
    print("=" * 75)
    print("✨ PROJETO EXECUTADO COM SUCESSO!")
    print("=" * 75)

if __name__ == "__main__":
    executar_projeto()
