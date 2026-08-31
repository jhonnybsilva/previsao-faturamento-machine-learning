"""
=============================================================================
PROJETO: ANÁLISE DE VENDAS + PREVISÃO DE FATURAMENTO COM MACHINE LEARNING
MÓDULO: GERADOR DE BANCO DE DADOS RELACIONAL E EXPORTAÇÃO NUVEM
AUTOR: Jhonny Brasiliano da Silva
=============================================================================
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def gerar_dados():
    print("[1/4] Gerando base de dados sintetica realista...")
    np.random.seed(42)
    
    # 1. Definição de Categorias e Produtos
    produtos_data = [
        {"id": 1, "nome": "Notebook Dell Inspiron 15", "categoria": "Eletronicos & TI", "preco": 3899.00, "custo": 2600.00},
        {"id": 2, "nome": "Monitor UltraWide 29 LG", "categoria": "Eletronicos & TI", "preco": 1250.00, "custo": 820.00},
        {"id": 3, "nome": "Teclado Mecanico RGB", "categoria": "Perifericos & Acessorios", "preco": 349.90, "custo": 180.00},
        {"id": 4, "nome": "Mouse Sem Fio Ergonomico", "categoria": "Perifericos & Acessorios", "preco": 189.90, "custo": 95.00},
        {"id": 5, "nome": "Headset Gamer 7.1", "categoria": "Audio & Video", "preco": 429.00, "custo": 230.00},
        {"id": 6, "nome": "Cadeira Ergonomica Pro", "categoria": "Moveis & Escritorio", "preco": 1199.00, "custo": 680.00},
        {"id": 7, "nome": "Mesa Regulavel Eletrica", "categoria": "Moveis & Escritorio", "preco": 2199.00, "custo": 1350.00},
        {"id": 8, "nome": "Webcam Full HD 1080p", "categoria": "Audio & Video", "preco": 279.90, "custo": 140.00},
        {"id": 9, "nome": "SSD NVMe 1TB Kingston", "categoria": "Hardware & Armazenamento", "preco": 489.00, "custo": 310.00},
        {"id": 10, "nome": "Licenca Software Cloud BI (Anual)", "categoria": "Software & Servicos", "preco": 1490.00, "custo": 350.00}
    ]
    df_produtos = pd.DataFrame(produtos_data)

    # 2. Definição de Regiões e Canais
    regioes = ["Sao Paulo", "Rio de Janeiro", "Minas Gerais", "Parana", "Rio Grande do Sul", "Bahia", "Distrito Federal"]
    canais = ["E-commerce Proprio", "Marketplace", "Vendas B2B Corporativo", "App Mobile"]
    metodos_pagamento = ["Cartao de Credito", "PIX", "Boleto Bancario", "Faturamento B2B"]
    status_venda = ["Concluido", "Concluido", "Concluido", "Concluido", "Cancelado"]

    # 3. Gerando Histórico Temporal de Vendas (36 meses: Jan/2023 a Dez/2025 + Jan a Ago/2026)
    data_inicio = datetime(2023, 1, 1)
    data_fim = datetime(2026, 8, 30)
    dias_totais = (data_fim - data_inicio).days
    
    vendas_lista = []
    itens_venda_lista = []
    
    id_venda = 1
    id_item = 1

    for dia_idx in range(dias_totais):
        data_atual = data_inicio + timedelta(days=dia_idx)
        mes = data_atual.month
        dia_semana = data_atual.weekday()
        
        # Tendência de crescimento constante de faturamento
        fator_tendencia = 1.0 + (dia_idx / dias_totais) * 0.40
        
        # Sazonalidade realista de varejo e B2B
        sazonalidade_mes = {
            1: 0.85,
            2: 0.88,
            3: 0.98,
            4: 1.02,
            5: 1.15,
            6: 1.08,
            7: 1.04,
            8: 1.12,
            9: 1.10,
            10: 1.18,
            11: 1.65, # Black Friday
            12: 1.78  # Natal
        }.get(mes, 1.0)
        
        fator_dia_semana = 1.20 if dia_semana in [1, 2, 3, 4] else 0.85
        
        base_vendas_dia = int(np.random.poisson(lam=16) * fator_tendencia * sazonalidade_mes * fator_dia_semana)
        base_vendas_dia = max(4, base_vendas_dia)
        
        for _ in range(base_vendas_dia):
            status = np.random.choice(status_venda, p=[0.90, 0.04, 0.03, 0.02, 0.01])
            regiao = np.random.choice(regioes, p=[0.40, 0.18, 0.14, 0.10, 0.08, 0.05, 0.05])
            canal = np.random.choice(canais, p=[0.45, 0.30, 0.15, 0.10])
            pagamento = np.random.choice(metodos_pagamento, p=[0.55, 0.30, 0.10, 0.05])
            
            num_itens = int(np.random.choice([1, 2, 3], p=[0.70, 0.22, 0.08]))
            produtos_escolhidos = df_produtos.sample(n=num_itens, replace=False)
            
            valor_total_venda = 0.0
            custo_total_venda = 0.0
            
            for _, prod in produtos_escolhidos.iterrows():
                qtd = int(np.random.choice([1, 2, 3, 5], p=[0.80, 0.14, 0.04, 0.02]))
                preco_unit = float(prod["preco"])
                custo_unit = float(prod["custo"])
                subtotal = qtd * preco_unit
                subtotal_custo = qtd * custo_unit
                
                valor_total_venda += subtotal
                custo_total_venda += subtotal_custo
                
                itens_venda_lista.append({
                    "id_item": id_item,
                    "id_venda": id_venda,
                    "id_produto": prod["id"],
                    "nome_produto": prod["nome"],
                    "categoria": prod["categoria"],
                    "quantidade": qtd,
                    "preco_unitario": preco_unit,
                    "custo_unitario": custo_unit,
                    "subtotal": subtotal,
                    "lucro_bruto": subtotal - subtotal_custo
                })
                id_item += 1
                
            desconto = 0.0
            if np.random.rand() < 0.25:
                desconto = round(valor_total_venda * np.random.uniform(0.05, 0.12), 2)
                
            valor_liquido = round(valor_total_venda - desconto, 2)
            
            vendas_lista.append({
                "id_venda": id_venda,
                "data_venda": data_atual.strftime("%Y-%m-%d"),
                "ano": data_atual.year,
                "mes": data_atual.month,
                "ano_mes": data_atual.strftime("%Y-%m"),
                "dia_semana": data_atual.strftime("%A"),
                "regiao": regiao,
                "canal": canal,
                "metodo_pagamento": pagamento,
                "status": status,
                "valor_bruto": valor_total_venda,
                "desconto": desconto,
                "valor_liquido": valor_liquido,
                "custo_total": custo_total_venda,
                "lucro_liquido": round(valor_liquido - custo_total_venda, 2)
            })
            id_venda += 1

    df_vendas = pd.DataFrame(vendas_lista)
    df_itens = pd.DataFrame(itens_venda_lista)
    
    print(f"[OK] Total de vendas geradas: {len(df_vendas):,} pedidos")
    print(f"[OK] Total de itens de vendas: {len(df_itens):,} itens")
    print(f"[OK] Faturamento acumulado no periodo: R$ {df_vendas['valor_liquido'].sum():,.2f}")
    
    # 4. Criando e Populando Banco SQLite Local
    print("\n[2/4] Criando banco de dados SQLite local (vendas_empresa.db)...")
    db_path = "vendas_empresa.db"
    conn = sqlite3.connect(db_path)
    
    df_produtos.to_sql("produtos", conn, if_exists="replace", index=False)
    df_vendas.to_sql("vendas", conn, if_exists="replace", index=False)
    df_itens.to_sql("itens_venda", conn, if_exists="replace", index=False)
    
    conn.commit()
    conn.close()
    print(f"[OK] Banco SQLite '{db_path}' criado com sucesso!")

    # 5. Exportando para CSV
    print("\n[3/4] Exportando dataset completo para CSV...")
    df_vendas_completas = df_itens.merge(df_vendas, on="id_venda", suffixes=('', '_venda'))
    df_vendas_completas.to_csv("vendas_historico_completo.csv", index=False, encoding="utf-8-sig")
    print("[OK] Arquivo 'vendas_historico_completo.csv' gerado.")

    # 6. Gerando Script SQL para Nuvem (Supabase / phpMyAdmin)
    print("\n[4/4] Gerando dump SQL compativel com Supabase e phpMyAdmin...")
    sql_path = "dump_banco_nuvem_supabase_mysql.sql"
    
    with open(sql_path, "w", encoding="utf-8") as f:
        f.write("-- ========================================================\n")
        f.write("-- BANCO DE DADOS: VENDAS & FATURAMENTO EMPRESARIAL\n")
        f.write("-- Compativel com: Supabase (PostgreSQL) e phpMyAdmin (MySQL)\n")
        f.write("-- Autor: Jhonny Brasiliano da Silva\n")
        f.write("-- ========================================================\n\n")
        
        # Tabela Produtos
        f.write("CREATE TABLE IF NOT EXISTS produtos (\n")
        f.write("    id INTEGER PRIMARY KEY,\n")
        f.write("    nome VARCHAR(100) NOT NULL,\n")
        f.write("    categoria VARCHAR(50) NOT NULL,\n")
        f.write("    preco DECIMAL(10,2) NOT NULL,\n")
        f.write("    custo DECIMAL(10,2) NOT NULL\n")
        f.write(");\n\n")
        
        for _, row in df_produtos.iterrows():
            f.write(f"INSERT INTO produtos (id, nome, categoria, preco, custo) VALUES ({row['id']}, '{row['nome']}', '{row['categoria']}', {row['preco']}, {row['custo']});\n")
        
        f.write("\n-- Tabela Vendas\n")
        f.write("CREATE TABLE IF NOT EXISTS vendas (\n")
        f.write("    id_venda INTEGER PRIMARY KEY,\n")
        f.write("    data_venda DATE NOT NULL,\n")
        f.write("    ano INTEGER,\n")
        f.write("    mes INTEGER,\n")
        f.write("    ano_mes VARCHAR(7),\n")
        f.write("    regiao VARCHAR(50),\n")
        f.write("    canal VARCHAR(50),\n")
        f.write("    metodo_pagamento VARCHAR(50),\n")
        f.write("    status VARCHAR(20),\n")
        f.write("    valor_bruto DECIMAL(10,2),\n")
        f.write("    desconto DECIMAL(10,2),\n")
        f.write("    valor_liquido DECIMAL(10,2),\n")
        f.write("    lucro_liquido DECIMAL(10,2)\n")
        f.write(");\n\n")
        
        # Amostra de inserts para o dump
        f.write("-- Amostra inicial de registros de vendas:\n")
        for _, row in df_vendas.head(200).iterrows():
            f.write(f"INSERT INTO vendas (id_venda, data_venda, ano, mes, ano_mes, regiao, canal, metodo_pagamento, status, valor_bruto, desconto, valor_liquido, lucro_liquido) "
                    f"VALUES ({row['id_venda']}, '{row['data_venda']}', {row['ano']}, {row['mes']}, '{row['ano_mes']}', '{row['regiao']}', '{row['canal']}', '{row['metodo_pagamento']}', '{row['status']}', {row['valor_bruto']}, {row['desconto']}, {row['valor_liquido']}, {row['lucro_liquido']});\n")
                    
    print(f"[OK] Arquivo SQL '{sql_path}' exportado com sucesso!")
    print("\n Base de dados pronta para analise e machine learning!")

if __name__ == "__main__":
    gerar_dados()
