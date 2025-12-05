"""
=============================================================================
PROJETO FINAL: SEGMENTAÇÃO DE CLIENTES COM MACHINE LEARNING CLÁSSICO
Análise de Personalidade de Clientes usando Clustering
=============================================================================
Autor: [Luiz Arthur, Lucas Oliverio]
Dataset: Customer Personality Analysis (Kaggle)
Objetivo: Segmentar clientes para otimização de campanhas de marketing
=============================================================================
"""

# =============================================================================
# 1. IMPORTAÇÃO DE BIBLIOTECAS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Visualizações
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 80)
print("SEGMENTAÇÃO DE CLIENTES - CUSTOMER PERSONALITY ANALYSIS")
print("=" * 80)

# =============================================================================
# 2. CARREGAMENTO DOS DADOS
# =============================================================================
print("\n[ETAPA 1] CARREGAMENTO DOS DADOS")
print("-" * 80)

# Baixar o dataset do Kaggle (você precisa ter o arquivo marketing_campaign.csv)
# Link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis
try:
    df = pd.read_csv('marketing_campaign.csv', sep='\t')
    print(f"✓ Dataset carregado com sucesso!")
    print(f"  - Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
except FileNotFoundError:
    print("⚠ ATENÇÃO: Arquivo 'marketing_campaign.csv' não encontrado!")
    print("  Por favor, baixe o dataset do Kaggle e coloque na mesma pasta do script.")
    print("  Link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis")
    exit()

# =============================================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# =============================================================================
print("\n[ETAPA 2] ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
print("-" * 80)

# 3.1 Informações Gerais
print("\n3.1 INFORMAÇÕES GERAIS DO DATASET")
print(df.info())

# 3.2 Estatísticas Descritivas
print("\n3.2 ESTATÍSTICAS DESCRITIVAS")
print(df.describe())

# 3.3 Valores Ausentes
print("\n3.3 ANÁLISE DE VALORES AUSENTES")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Coluna': missing_values.index,
    'Valores Ausentes': missing_values.values,
    'Percentual (%)': missing_percent.values
})
missing_df = missing_df[missing_df['Valores Ausentes'] > 0].sort_values('Valores Ausentes', ascending=False)
print(missing_df)

# 3.4 Visualização de Valores Ausentes
if len(missing_df) > 0:
    plt.figure(figsize=(10, 5))
    plt.bar(missing_df['Coluna'], missing_df['Percentual (%)'], color='coral')
    plt.xlabel('Colunas')
    plt.ylabel('Percentual de Valores Ausentes (%)')
    plt.title('Valores Ausentes por Coluna')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('01_valores_ausentes.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico salvo: 01_valores_ausentes.png")
    plt.close()

# 3.5 Análise de Duplicatas
duplicates = df.duplicated().sum()
print(f"\n3.5 DUPLICATAS: {duplicates} linhas duplicadas encontradas")

# =============================================================================
# 4. PRÉ-PROCESSAMENTO E FEATURE ENGINEERING
# =============================================================================
print("\n[ETAPA 3] PRÉ-PROCESSAMENTO E FEATURE ENGINEERING")
print("-" * 80)

# Criar cópia para manipulação
df_processed = df.copy()

# 4.1 Tratamento de Valores Ausentes
print("\n4.1 TRATAMENTO DE VALORES AUSENTES")
# Income tem valores ausentes - preencher com a mediana
if df_processed['Income'].isnull().sum() > 0:
    median_income = df_processed['Income'].median()
    df_processed['Income'].fillna(median_income, inplace=True)
    print(f"✓ Valores ausentes em 'Income' preenchidos com a mediana: {median_income:.2f}")

# 4.2 Feature Engineering - Criação de Novas Variáveis
print("\n4.2 FEATURE ENGINEERING - CRIAÇÃO DE NOVAS VARIÁVEIS")

# Idade do cliente
current_year = 2025
df_processed['Age'] = current_year - df_processed['Year_Birth']
print(f"✓ Variável 'Age' criada (idade do cliente)")

# Total de gastos
spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df_processed['Total_Spending'] = df_processed[spending_columns].sum(axis=1)
print(f"✓ Variável 'Total_Spending' criada (gasto total)")

# Total de filhos
df_processed['Total_Children'] = df_processed['Kidhome'] + df_processed['Teenhome']
print(f"✓ Variável 'Total_Children' criada (total de filhos)")

# Total de compras
purchase_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df_processed['Total_Purchases'] = df_processed[purchase_columns].sum(axis=1)
print(f"✓ Variável 'Total_Purchases' criada (total de compras)")

# Total de campanhas aceitas
campaign_columns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                   'AcceptedCmp4', 'AcceptedCmp5', 'Response']
df_processed['Total_Campaigns_Accepted'] = df_processed[campaign_columns].sum(axis=1)
print(f"✓ Variável 'Total_Campaigns_Accepted' criada (campanhas aceitas)")

# Tempo como cliente (em dias)
df_processed['Dt_Customer'] = pd.to_datetime(df_processed['Dt_Customer'], format='%d-%m-%Y')
reference_date = df_processed['Dt_Customer'].max()
df_processed['Customer_Days'] = (reference_date - df_processed['Dt_Customer']).dt.days
print(f"✓ Variável 'Customer_Days' criada (dias como cliente)")

# Gasto médio por compra
df_processed['Avg_Spending_Per_Purchase'] = df_processed['Total_Spending'] / (df_processed['Total_Purchases'] + 1)
print(f"✓ Variável 'Avg_Spending_Per_Purchase' criada (gasto médio por compra)")

# 4.3 Codificação de Variáveis Categóricas
print("\n4.3 CODIFICAÇÃO DE VARIÁVEIS CATEGÓRICAS")

# Education - Label Encoding ordenado
education_mapping = {
    'Basic': 1,
    '2n Cycle': 2,
    'Graduation': 3,
    'Master': 4,
    'PhD': 5
}
df_processed['Education_Encoded'] = df_processed['Education'].map(education_mapping)
print(f"✓ 'Education' codificada (1=Basic até 5=PhD)")

# Marital Status - Simplificar e codificar
marital_mapping = {
    'Single': 0,
    'Together': 1,
    'Married': 1,
    'Divorced': 0,
    'Widow': 0,
    'Alone': 0,
    'Absurd': 0,
    'YOLO': 0
}
df_processed['Is_Partnered'] = df_processed['Marital_Status'].map(marital_mapping)
print(f"✓ 'Marital_Status' simplificada para 'Is_Partnered' (0=Sozinho, 1=Acompanhado)")

# 4.4 Remoção de Outliers Extremos
print("\n4.4 TRATAMENTO DE OUTLIERS")
# Remover idades irreais (ex: > 100 anos ou < 18 anos)
initial_rows = len(df_processed)
df_processed = df_processed[(df_processed['Age'] >= 18) & (df_processed['Age'] <= 100)]
removed_rows = initial_rows - len(df_processed)
print(f"✓ Removidas {removed_rows} linhas com idades fora do intervalo [18, 100]")

# Remover rendas extremamente altas (outliers)
income_q99 = df_processed['Income'].quantile(0.99)
df_processed = df_processed[df_processed['Income'] <= income_q99]
print(f"✓ Removidos outliers de renda acima do percentil 99 (>{income_q99:.2f})")

# 4.5 Seleção de Features para Clustering
print("\n4.5 SELEÇÃO DE FEATURES PARA CLUSTERING")
features_for_clustering = [
    'Age',
    'Income',
    'Total_Spending',
    'Total_Children',
    'Total_Purchases',
    'Total_Campaigns_Accepted',
    'Customer_Days',
    'Recency',
    'NumWebVisitsMonth',
    'Education_Encoded',
    'Is_Partnered',
    'Avg_Spending_Per_Purchase'
]

X = df_processed[features_for_clustering].copy()
print(f"✓ {len(features_for_clustering)} features selecionadas para clustering")
print(f"  Features: {', '.join(features_for_clustering)}")

# 4.6 Normalização dos Dados
print("\n4.6 NORMALIZAÇÃO DOS DADOS (StandardScaler)")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features_for_clustering, index=X.index)
print(f"✓ Dados normalizados com StandardScaler (média=0, desvio=1)")

# =============================================================================
# 5. VISUALIZAÇÕES DA EDA
# =============================================================================
print("\n[ETAPA 4] VISUALIZAÇÕES DA ANÁLISE EXPLORATÓRIA")
print("-" * 80)

# 5.1 Distribuição de Idade
plt.figure(figsize=(14, 10))

plt.subplot(3, 3, 1)
plt.hist(df_processed['Age'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.title('Distribuição de Idade dos Clientes')

# 5.2 Distribuição de Renda
plt.subplot(3, 3, 2)
plt.hist(df_processed['Income'], bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Renda Anual')
plt.ylabel('Frequência')
plt.title('Distribuição de Renda')

# 5.3 Total de Gastos
plt.subplot(3, 3, 3)
plt.hist(df_processed['Total_Spending'], bins=30, color='coral', edgecolor='black')
plt.xlabel('Gasto Total')
plt.ylabel('Frequência')
plt.title('Distribuição de Gastos Totais')

# 5.4 Educação
plt.subplot(3, 3, 4)
education_counts = df_processed['Education'].value_counts()
plt.bar(education_counts.index, education_counts.values, color='plum')
plt.xlabel('Nível de Educação')
plt.ylabel('Quantidade')
plt.title('Distribuição por Educação')
plt.xticks(rotation=45, ha='right')

# 5.5 Estado Civil
plt.subplot(3, 3, 5)
marital_counts = df_processed['Marital_Status'].value_counts()
plt.bar(marital_counts.index, marital_counts.values, color='gold')
plt.xlabel('Estado Civil')
plt.ylabel('Quantidade')
plt.title('Distribuição por Estado Civil')
plt.xticks(rotation=45, ha='right')

# 5.6 Total de Filhos
plt.subplot(3, 3, 6)
children_counts = df_processed['Total_Children'].value_counts().sort_index()
plt.bar(children_counts.index, children_counts.values, color='lightblue')
plt.xlabel('Número de Filhos')
plt.ylabel('Quantidade')
plt.title('Distribuição de Filhos')

# 5.7 Campanhas Aceitas
plt.subplot(3, 3, 7)
campaigns_counts = df_processed['Total_Campaigns_Accepted'].value_counts().sort_index()
plt.bar(campaigns_counts.index, campaigns_counts.values, color='salmon')
plt.xlabel('Campanhas Aceitas')
plt.ylabel('Quantidade')
plt.title('Distribuição de Campanhas Aceitas')

# 5.8 Correlação Renda vs Gastos
plt.subplot(3, 3, 8)
plt.scatter(df_processed['Income'], df_processed['Total_Spending'], alpha=0.5, color='purple')
plt.xlabel('Renda')
plt.ylabel('Gasto Total')
plt.title('Renda vs Gasto Total')

# 5.9 Idade vs Gastos
plt.subplot(3, 3, 9)
plt.scatter(df_processed['Age'], df_processed['Total_Spending'], alpha=0.5, color='teal')
plt.xlabel('Idade')
plt.ylabel('Gasto Total')
plt.title('Idade vs Gasto Total')

plt.tight_layout()
plt.savefig('02_eda_visualizacoes.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 02_eda_visualizacoes.png")
plt.close()

# 5.10 Matriz de Correlação
plt.figure(figsize=(14, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlação das Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_matriz_correlacao.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 03_matriz_correlacao.png")
plt.close()

# =============================================================================
# 6. DETERMINAÇÃO DO NÚMERO ÓTIMO DE CLUSTERS
# =============================================================================
print("\n[ETAPA 5] DETERMINAÇÃO DO NÚMERO ÓTIMO DE CLUSTERS")
print("-" * 80)

# 6.1 Método do Cotovelo (Elbow Method)
print("\n6.1 MÉTODO DO COTOVELO (ELBOW METHOD)")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Visualização do Método do Cotovelo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de Inércia
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
ax1.set_ylabel('Inércia (Within-Cluster Sum of Squares)', fontsize=12)
ax1.set_title('Método do Cotovelo', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Gráfico de Silhouette Score
ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Análise de Silhouette Score', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_elbow_silhouette.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico salvo: 04_elbow_silhouette.png")
plt.close()

# Determinar K ótimo baseado no Silhouette Score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✓ NÚMERO ÓTIMO DE CLUSTERS SUGERIDO: K = {optimal_k}")
print(f"  (Baseado no maior Silhouette Score: {max(silhouette_scores):.4f})")

# =============================================================================
# 7. TREINAMENTO DOS MODELOS DE CLUSTERING
# =============================================================================
print("\n[ETAPA 6] TREINAMENTO DOS MODELOS DE CLUSTERING")
print("-" * 80)

# Usar K ótimo determinado
K_OPTIMAL = optimal_k

# 7.1 MODELO A: K-Means
print(f"\n7.1 MODELO A: K-MEANS (K={K_OPTIMAL})")
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(X_scaled)
df_processed['Cluster_KMeans'] = kmeans_labels

# Métricas K-Means
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"✓ K-Means treinado com sucesso!")
print(f"  - Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  - Davies-Bouldin Index: {kmeans_davies_bouldin:.4f} (menor é melhor)")
print(f"  - Calinski-Harabasz Score: {kmeans_calinski:.2f} (maior é melhor)")

# 7.2 MODELO B: Hierarchical Clustering (Agglomerative)
print(f"\n7.2 MODELO B: HIERARCHICAL CLUSTERING (K={K_OPTIMAL})")
hierarchical = AgglomerativeClustering(n_clusters=K_OPTIMAL, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)
df_processed['Cluster_Hierarchical'] = hierarchical_labels

# Métricas Hierarchical
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hierarchical_davies_bouldin = davies_bouldin_score(X_scaled, hierarchical_labels)
hierarchical_calinski = calinski_harabasz_score(X_scaled, hierarchical_labels)

print(f"✓ Hierarchical Clustering treinado com sucesso!")
print(f"  - Silhouette Score: {hierarchical_silhouette:.4f}")
print(f"  - Davies-Bouldin Index: {hierarchical_davies_bouldin:.4f}")
print(f"  - Calinski-Harabasz Score: {hierarchical_calinski:.2f}")

# 7.3 MODELO C: DBSCAN
print(f"\n7.3 MODELO C: DBSCAN (Densidade)")
dbscan = DBSCAN(eps=3.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)
df_processed['Cluster_DBSCAN'] = dbscan_labels

# Contar clusters (excluindo ruído = -1)
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"✓ DBSCAN treinado com sucesso!")
print(f"  - Clusters encontrados: {n_clusters_dbscan}")
print(f"  - Pontos de ruído: {n_noise}")

if n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print(f"  - Silhouette Score: {dbscan_silhouette:.4f}")
else:
    print(f"  - Silhouette Score: N/A (apenas 1 cluster)")

# =============================================================================
# 8. COMPARAÇÃO DE MODELOS
# =============================================================================
print("\n[ETAPA 7] COMPARAÇÃO DE MODELOS")
print("-" * 80)

# Tabela de Comparação
comparison_df = pd.DataFrame({
    'Modelo': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, hierarchical_silhouette, 
                         dbscan_silhouette if n_clusters_dbscan > 1 else np.nan],
    'Davies-Bouldin Index': [kmeans_davies_bouldin, hierarchical_davies_bouldin, np.nan],
    'Calinski-Harabasz Score': [kmeans_calinski, hierarchical_calinski, np.nan],
    'Número de Clusters': [K_OPTIMAL, K_OPTIMAL, n_clusters_dbscan]
})

print("\nTABELA DE COMPARAÇÃO DE MODELOS:")
print("=" * 100)
print(comparison_df.to_string(index=False))
print("=" * 100)

# Determinar melhor modelo
best_model_idx = comparison_df['Silhouette Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Modelo']
print(f"\n✓ MELHOR MODELO: {best_model_name}")
print(f"  (Baseado no maior Silhouette Score)")

# Salvar tabela
comparison_df.to_csv('05_comparacao_modelos.csv', index=False)
print("\n✓ Tabela salva: 05_comparacao_modelos.csv")

# =============================================================================
# 9. VISUALIZAÇÃO DOS CLUSTERS (PCA)
# =============================================================================
print("\n[ETAPA 8] VISUALIZAÇÃO DOS CLUSTERS COM PCA")
print("-" * 80)

# Redução de dimensionalidade com PCA (2 componentes)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA aplicado: {pca.explained_variance_ratio_[0]*100:.2f}% + "
      f"{pca.explained_variance_ratio_[1]*100:.2f}% = "
      f"{sum(pca.explained_variance_ratio_)*100:.2f}% da variância explicada")

# Visualização dos 3 modelos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, 
                          cmap='viridis', alpha=0.6, edgecolors='k', s=50)
axes[0].scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
               pca.transform(kmeans.cluster_centers_)[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroides')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title(f'K-Means (Silhouette: {kmeans_silhouette:.3f})', fontweight='bold')
axes[0].legend()
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Hierarchical
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, 
                          cmap='plasma', alpha=0.6, edgecolors='k', s=50)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].set_title(f'Hierarchical (Silhouette: {hierarchical_silhouette:.3f})', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

# DBSCAN
scatter3 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, 
                          cmap='coolwarm', alpha=0.6, edgecolors='k', s=50)
axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].set_title(f'DBSCAN ({n_clusters_dbscan} clusters, {n_noise} ruídos)', fontweight='bold')
plt.colorbar(scatter3, ax=axes[2], label='Cluster')

plt.tight_layout()
plt.savefig('06_clusters_pca.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 06_clusters_pca.png")
plt.close()

# =============================================================================
# 10. ANÁLISE E INTERPRETAÇÃO DOS CLUSTERS (K-MEANS)
# =============================================================================
print("\n[ETAPA 9] ANÁLISE E INTERPRETAÇÃO DOS CLUSTERS (K-MEANS)")
print("-" * 80)

# Estatísticas por cluster
cluster_analysis = df_processed.groupby('Cluster_KMeans').agg({
    'Age': 'mean',
    'Income': 'mean',
    'Total_Spending': 'mean',
    'Total_Children': 'mean',
    'Total_Purchases': 'mean',
    'Total_Campaigns_Accepted': 'mean',
    'Education_Encoded': 'mean',
    'Recency': 'mean',
    'Customer_Days': 'mean',
    'ID': 'count'  # Contagem de clientes
}).round(2)

cluster_analysis.rename(columns={'ID': 'Num_Clientes'}, inplace=True)

print("\nESTATÍSTICAS POR CLUSTER:")
print("=" * 120)
print(cluster_analysis)
print("=" * 120)

# Salvar análise
cluster_analysis.to_csv('07_analise_clusters.csv')
print("\n✓ Análise salva: 07_analise_clusters.csv")

# Interpretação dos Clusters
print("\n" + "=" * 80)
print("INTERPRETAÇÃO DOS CLUSTERS (PERFIS DE CLIENTES)")
print("=" * 80)

for cluster_id in range(K_OPTIMAL):
    cluster_data = cluster_analysis.loc[cluster_id]
    print(f"\n🔹 CLUSTER {cluster_id} - {int(cluster_data['Num_Clientes'])} clientes "
          f"({cluster_data['Num_Clientes']/len(df_processed)*100:.1f}%)")
    print(f"   • Idade média: {cluster_data['Age']:.1f} anos")
    print(f"   • Renda média: R$ {cluster_data['Income']:.2f}")
    print(f"   • Gasto total médio: R$ {cluster_data['Total_Spending']:.2f}")
    print(f"   • Filhos (média): {cluster_data['Total_Children']:.2f}")
    print(f"   • Compras totais (média): {cluster_data['Total_Purchases']:.1f}")
    print(f"   • Campanhas aceitas (média): {cluster_data['Total_Campaigns_Accepted']:.2f}")
    print(f"   • Educação (1-5): {cluster_data['Education_Encoded']:.2f}")
    print(f"   • Recência (dias): {cluster_data['Recency']:.1f}")
    
    # Sugestão de perfil
    if cluster_data['Total_Spending'] > cluster_analysis['Total_Spending'].mean():
        if cluster_data['Income'] > cluster_analysis['Income'].mean():
            perfil = "💎 CLIENTE PREMIUM (Alto valor)"
        else:
            perfil = "🛍️ COMPRADOR FREQUENTE"
    else:
        if cluster_data['Total_Campaigns_Accepted'] > cluster_analysis['Total_Campaigns_Accepted'].mean():
            perfil = "🎯 SENSÍVEL A PROMOÇÕES"
        else:
            perfil = "💤 CLIENTE INATIVO/ECONÔMICO"
    
    print(f"   ➜ Perfil sugerido: {perfil}")

# =============================================================================
# 11. FEATURE IMPORTANCE (Análise de Contribuição)
# =============================================================================
print("\n[ETAPA 10] ANÁLISE DE FEATURE IMPORTANCE")
print("-" * 80)

# Calcular a variância de cada feature dentro de cada cluster
feature_importance = []

for feature in features_for_clustering:
    # Variância entre clusters (quanto maior, mais importante)
    cluster_means = df_processed.groupby('Cluster_KMeans')[feature].mean()
    between_cluster_variance = cluster_means.var()
    feature_importance.append({
        'Feature': feature,
        'Between_Cluster_Variance': between_cluster_variance
    })

importance_df = pd.DataFrame(feature_importance).sort_values('Between_Cluster_Variance', ascending=False)
importance_df['Importance_Normalized'] = (importance_df['Between_Cluster_Variance'] / 
                                          importance_df['Between_Cluster_Variance'].sum() * 100)

print("\nFEATURE IMPORTANCE (Variância Entre Clusters):")
print("=" * 80)
print(importance_df.to_string(index=False))
print("=" * 80)

# Visualização
plt.figure(figsize=(12, 6))
plt.barh(importance_df['Feature'], importance_df['Importance_Normalized'], color='steelblue')
plt.xlabel('Importância Normalizada (%)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance - Contribuição para Separação dos Clusters', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('08_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfico salvo: 08_feature_importance.png")
plt.close()

# Salvar
importance_df.to_csv('08_feature_importance.csv', index=False)
print("✓ Tabela salva: 08_feature_importance.csv")

# =============================================================================
# 12. VISUALIZAÇÕES ADICIONAIS DOS CLUSTERS
# =============================================================================
print("\n[ETAPA 11] VISUALIZAÇÕES ADICIONAIS DOS CLUSTERS")
print("-" * 80)

# 12.1 Distribuição de clientes por cluster
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Contagem por cluster
cluster_counts = df_processed['Cluster_KMeans'].value_counts().sort_index()
axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='teal', edgecolor='black')
axes[0, 0].set_xlabel('Cluster')
axes[0, 0].set_ylabel('Número de Clientes')
axes[0, 0].set_title('Distribuição de Clientes por Cluster', fontweight='bold')
for i, v in enumerate(cluster_counts.values):
    axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Renda por cluster
df_processed.boxplot(column='Income', by='Cluster_KMeans', ax=axes[0, 1])
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Renda')
axes[0, 1].set_title('Distribuição de Renda por Cluster', fontweight='bold')
plt.sca(axes[0, 1])
plt.xticks(rotation=0)

# Gastos por cluster
df_processed.boxplot(column='Total_Spending', by='Cluster_KMeans', ax=axes[1, 0])
axes[1, 0].set_xlabel('Cluster')
axes[1, 0].set_ylabel('Gasto Total')
axes[1, 0].set_title('Distribuição de Gastos por Cluster', fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(rotation=0)

# Idade por cluster
df_processed.boxplot(column='Age', by='Cluster_KMeans', ax=axes[1, 1])
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Idade')
axes[1, 1].set_title('Distribuição de Idade por Cluster', fontweight='bold')
plt.sca(axes[1, 1])
plt.xticks(rotation=0)

plt.suptitle('')  # Remove título automático do boxplot
plt.tight_layout()
plt.savefig('09_clusters_distribuicoes.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 09_clusters_distribuicoes.png")
plt.close()

# 12.2 Heatmap de características por cluster
plt.figure(figsize=(12, 6))
cluster_heatmap = df_processed.groupby('Cluster_KMeans')[features_for_clustering].mean()
cluster_heatmap_normalized = (cluster_heatmap - cluster_heatmap.mean()) / cluster_heatmap.std()

sns.heatmap(cluster_heatmap_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Heatmap de Características por Cluster (Valores Normalizados)', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('10_clusters_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 10_clusters_heatmap.png")
plt.close()

# =============================================================================
# 13. DENDROGRAMA (Hierarchical Clustering)
# =============================================================================
print("\n[ETAPA 12] DENDROGRAMA (HIERARCHICAL CLUSTERING)")
print("-" * 80)

# Criar dendrograma com amostra (para performance)
sample_size = min(500, len(X_scaled))
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_indices]

plt.figure(figsize=(14, 6))
linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix, no_labels=True, color_threshold=0)
plt.xlabel('Índice da Amostra', fontsize=12)
plt.ylabel('Distância', fontsize=12)
plt.title(f'Dendrograma - Hierarchical Clustering (Amostra de {sample_size} clientes)', 
          fontsize=14, fontweight='bold')
plt.axhline(y=50, color='r', linestyle='--', label=f'Corte sugerido (K={K_OPTIMAL})')
plt.legend()
plt.tight_layout()
plt.savefig('11_dendrograma.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico salvo: 11_dendrograma.png")
plt.close()

# =============================================================================
# 14. EXPORTAÇÃO DOS RESULTADOS FINAIS
# =============================================================================
print("\n[ETAPA 13] EXPORTAÇÃO DOS RESULTADOS FINAIS")
print("-" * 80)

# Salvar dataset com clusters
output_columns = ['ID', 'Age', 'Income', 'Education', 'Marital_Status', 
                 'Total_Spending', 'Total_Children', 'Total_Purchases',
                 'Total_Campaigns_Accepted', 'Cluster_KMeans', 
                 'Cluster_Hierarchical', 'Cluster_DBSCAN']

df_output = df_processed[output_columns].copy()
df_output.to_csv('12_clientes_segmentados.csv', index=False)
print("✓ Dataset com clusters salvo: 12_clientes_segmentados.csv")

# =============================================================================
# 15. RELATÓRIO FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RELATÓRIO FINAL - SEGMENTAÇÃO DE CLIENTES")
print("=" * 80)

print(f"""
📊 RESUMO DO PROJETO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DATASET
   • Total de clientes analisados: {len(df_processed)}
   • Features utilizadas: {len(features_for_clustering)}
   • Período de análise: {df_processed['Dt_Customer'].min().strftime('%d/%m/%Y')} a {df_processed['Dt_Customer'].max().strftime('%d/%m/%Y')}

2. PRÉ-PROCESSAMENTO
   • Valores ausentes tratados: Sim (Income preenchida com mediana)
   • Outliers removidos: Sim (idade e renda extremas)
   • Normalização: StandardScaler
   • Feature Engineering: 7 novas variáveis criadas

3. MODELOS TREINADOS
   • Modelo A: K-Means (Silhouette: {kmeans_silhouette:.4f})
   • Modelo B: Hierarchical Clustering (Silhouette: {hierarchical_silhouette:.4f})
   • Modelo C: DBSCAN (Clusters: {n_clusters_dbscan}, Ruídos: {n_noise})

4. MELHOR MODELO
   • {best_model_name} (Maior Silhouette Score)
   • Número de clusters: {K_OPTIMAL}

5. PRINCIPAIS INSIGHTS
   • Clusters identificados representam perfis distintos de clientes
   • Features mais importantes: {importance_df.iloc[0]['Feature']}, {importance_df.iloc[1]['Feature']}, {importance_df.iloc[2]['Feature']}
   • Variância explicada pelo PCA (2D): {sum(pca.explained_variance_ratio_)*100:.2f}%

6. ARQUIVOS GERADOS
   ✓ 01_valores_ausentes.png
   ✓ 02_eda_visualizacoes.png
   ✓ 03_matriz_correlacao.png
   ✓ 04_elbow_silhouette.png
   ✓ 05_comparacao_modelos.csv
   ✓ 06_clusters_pca.png
   ✓ 07_analise_clusters.csv
   ✓ 08_feature_importance.png
   ✓ 08_feature_importance.csv
   ✓ 09_clusters_distribuicoes.png
   ✓ 10_clusters_heatmap.png
   ✓ 11_dendrograma.png
   ✓ 12_clientes_segmentados.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ PROJETO CONCLUÍDO COM SUCESSO!

📌 PRÓXIMOS PASSOS SUGERIDOS:
   1. Validar os perfis de clientes com especialistas de marketing
   2. Desenvolver estratégias de campanha específicas para cada cluster
   3. Implementar sistema de predição de cluster para novos clientes
   4. Monitorar a evolução dos clusters ao longo do tempo

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("\n" + "=" * 80)
print("FIM DO SCRIPT")
print("=" * 80)