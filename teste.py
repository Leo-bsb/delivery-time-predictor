import polars as pl


# ============================================
# 1. CARREGAMENTO E EXPLORAÇÃO DOS DADOS
# ============================================

# Carrega o dataset (ajuste o caminho)
df = pl.read_csv('/home/leonardo/scripts/delivery-predictor/Food_Delivery_Times.csv')
df.sample(2)