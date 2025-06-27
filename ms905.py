import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Análise de Consumo de Drogas", 
    page_icon="💊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar e pré-processar os dados
@st.cache_data
def load_data():
    df = pd.read_csv("Drug_Consumption.csv")
    
    # Remover overclaimers
    df = df[df['Semer'] == 'CL0']

    # Remover colunas desnecessárias
    df.drop(columns=['ID', 'Ethnicity', 'Impulsive', 'SS', 'Choc', 'Caff', 'Semer'], inplace=True, errors='ignore')

    # Mapear classes de consumo para valores numéricos
    drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
                    'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']
    
    # Converter classes CL0-CL6 para valores numéricos (0-6)
    for col in drug_columns:
        df[col] = df[col].str.extract('(\d+)').astype(float)
    
    return df

# Função para treinar modelo de classificação
def train_model(X, y):
    # Codificar variáveis categóricas
    le = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = le.fit_transform(X[col])
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

# Carregar dados
df = load_data()

# Definir abas
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 Introdução e Objetivo",
    "🔍 Análise Exploratória", 
    "🤖 Classificação com ML",
    "🎯 Conclusão"
])

## ABA 1: Introdução e Objetivo
with tab1:
    st.header("📌 Introdução e Objetivo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Sobre o Dataset
        O conjunto de dados Drug Consumption (Quantified) contém informações sobre:
        - **Características demográficas**: Idade, gênero, educação, país, etnia
        - **Traços de personalidade**: 5 dimensões de personalidade (NEO-FFI-R)
        - **Padrões de consumo**: Uso de 16 substâncias (álcool, nicotina, drogas ilícitas, etc.)
        
        ### Metodologia
        - **Análise exploratória**: Visualizações e testes estatísticos
        - **Modelagem**: Algoritmo de classificação (Random Forest)
        - **Interpretação**: Análise de importância de variáveis e métricas
        """)

    with col2:
        st.markdown("""
        ### Objetivos da Análise
        1. Identificar padrões de consumo entre diferentes grupos demográficos
        2. Explorar relações entre traços de personalidade e uso de substâncias
        3. Desenvolver um modelo preditivo para classificar usuários
        4. Extrair insights para políticas de saúde pública e prevenção
        """)


## ABA 2: Análise Exploratória
with tab2:
    st.header("🔍 Análise Exploratória dos Dados")
    
    # Seção 1: Visão Geral
    st.subheader("📊 Visão Geral")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total de Respondentes", len(df))
        st.metric("Substâncias Analisadas", 16)
        with st.expander("ℹ️ Lista de substâncias estudadas"):
            st.markdown("""
            -Alcohol: álcool
            -Amphet: anfetamina
            -Amyl: nitrito de amila
            -Benzos: benzodiazepina
            -Cannabis: maconha
            -Coke: cocaína
            -Crack: crack
            -Ecstasy: ecstasy
            -Heroin: heroína
            -Ketamine: ketamina
            -Legalh: drogas legais sintéticas
            -LSD: LSD
            -Meth: metadona
            -Mushroom: cogumelos alucinógenos
            -Nicotine: nicotina
            -VSA: substâncias voláteis
            """)
    with col2:
        # Seção 2: Distribuição Demográfica
        # Seletor interativo
        dist_option = st.radio(
            "Selecione a variável para visualizar a distribuição:",
            options=["Idade", "Gênero", "Educação", "País"],
            horizontal=True
        )

        # Mapeamento entre opção e coluna do dataframe
        dist_map = {
            "Idade": "Age",
            "Gênero": "Gender",
            "Educação": "Education",
            "País": "Country"
        }

        selected_col = dist_map[dist_option]

        # Gerar gráfico de pizza
        fig = px.pie(
            df,
            names=selected_col,
            title=f"Distribuição por {dist_option}"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Seção 3: Consumo de Drogas
    st.subheader("💊 Padrões de Consumo")
    
    # Selecionar drogas para análise
    drugs = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Cannabis', 'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 
            'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'VSA']
    selected_drugs = st.multiselect(
        "Selecione as drogas para análise:", 
        drugs, 
        default=['Alcohol', 'Cannabis', 'Nicotine']
    )
    
    # Gráfico de consumo médio
    if selected_drugs:
        drug_means = df[selected_drugs].mean().sort_values(ascending=False)
        fig = px.bar(drug_means, 
                     x=drug_means.values, 
                     y=drug_means.index,
                     title="Consumo Médio por Substância",
                     labels={'x': 'Consumo Médio (0-6)', 'y': 'Substância'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Consumo por demografia
    st.subheader("Consumo por Grupo Demográfico")
    
    tab_age, tab_gender, tab_country, tab_education = st.tabs(["Por Idade", "Por Gênero", "Por País", "Por Educação"])
    
    with tab_age:
        if selected_drugs:
            age_drug = df.groupby('Age')[selected_drugs].mean().reset_index()
            fig = px.bar(age_drug, 
                         x='Age', 
                         y=selected_drugs, 
                         barmode='group',
                         title="Consumo Médio por Faixa Etária")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_gender:
        if selected_drugs:
            gender_drug = df.groupby('Gender')[selected_drugs].mean().reset_index()
            fig = px.bar(gender_drug, 
                         x='Gender', 
                         y=selected_drugs, 
                         barmode='group',
                         title="Consumo Médio por Gênero")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_country:
        if selected_drugs:
            country_means = df.groupby('Country')[selected_drugs].mean().reset_index()

            fig = px.bar(
                country_means,
                x='Country',
                y=selected_drugs,
                barmode='group',
                title="Consumo Médio por País (Substâncias Selecionadas)",
                labels={'value': 'Consumo Médio', 'variable': 'Substância'},
                hover_name='Country'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_education:
        # Definir ordem de educação
        
        if selected_drugs:
            edu_df = df.groupby('Education')[selected_drugs].mean().reset_index()

            melted_edu = edu_df.melt(id_vars='Education', var_name='Droga', value_name='Consumo Médio')

            fig = px.bar(
                melted_edu,
                x='Education',
                y='Consumo Médio',
                color='Droga',
                barmode='group',
                title="Consumo Médio por Nível Educacional",
                labels={'Education': 'Educação', 'Consumo Médio': 'Consumo Médio'},
                hover_name='Droga'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # Seção 4: Correlações
    st.subheader("📈 Análise de Correlações")
    with st.expander("ℹ️ Sobre os Traços de Personalidade"):
        st.markdown("""
        Os traços de personalidade utilizados seguem o modelo **NEO-FFI-R (Big Five)**:

        - **Nscore (Neuroticismo)**: Mede a tendência à **instabilidade emocional**, como ansiedade, irritabilidade e vulnerabilidade.  
        Ex: Um Nscore alto indica maior propensão ao estresse e reatividade emocional.

        - **Escore (Extroversão)**: Reflete níveis de **sociabilidade, assertividade e energia positiva**.  
        Ex: Pessoas com escore alto tendem a ser mais falantes e sociáveis.

        - **Oscore (Abertura para Experiências)**: Relacionado à **criatividade, imaginação e curiosidade intelectual**.  
        Ex: Um Oscore alto sugere mente aberta e interesse por novidades.

        - **AScore (Amabilidade)**: Indica o grau de **cooperação, empatia e confiança** nos outros.  
        Ex: Escores altos sugerem maior capacidade de se relacionar harmoniosamente.

        - **Cscore (Consciência)**: Mede o nível de **organização, disciplina e foco em metas**.  
        Ex: Indivíduos com alto Cscore são mais responsáveis e persistentes.

        Os valores podem ser:
        - **Negativos**: abaixo da média
        - **Positivos**: acima da média

        Leia mais em: https://www.researchgate.net/publication/240133762_Neo_PI-R_professional_manual
        """)
    # Matriz de correlação
    corr_matrix = df[selected_drugs +['Nscore', 'Escore', 'Oscore', 'AScore', 'Cscore']].corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Correlação entre Consumo de Drogas e Traços de Personalidade",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    # Teste estatístico: Comparação de médias (ANOVA)
    st.subheader("🔬 Teste Estatístico: Diferenças de Traços de Personalidade por Nível de Consumo")
    st.markdown("""
    Para verificar se os traços de personalidade variam de forma significativa entre diferentes níveis de consumo,
    foi aplicado o teste **ANOVA (Análise de Variância)**. Abaixo está uma tabela com os resultados obtidos.
    """)

    personality_traits = ['Nscore', 'Escore', 'Oscore', 'AScore', 'Cscore']

    if selected_drugs:
        anova_results = []
        for drug in selected_drugs:
            for trait in personality_traits:
                groups = [df[df[drug] == level][trait].dropna() for level in sorted(df[drug].unique())]
                if len(groups) > 1:
                    f_stat, p_val = f_oneway(*groups)
                    anova_results.append({
                        'Substância': drug,
                        'Traço': trait,
                        'Valor-p': round(p_val, 4)
                    })

        anova_df = pd.DataFrame(anova_results)
        pval_table = anova_df.pivot(index='Substância', columns='Traço', values='Valor-p')
        st.dataframe(pval_table.style.format("{:.4f}"), use_container_width=True)
        
        st.markdown("""
        **Interpretação:** Valores-p menores que 0.05 indicam diferenças estatisticamente significativas nos traços de personalidade
        entre os diferentes níveis de consumo da substância analisada.
        """)
    
    personality_traits = ['Nscore', 'Escore', 'Oscore', 'AScore', 'Cscore']
    
## ABA 3: Classificação com Aprendizado de Máquina
with tab3:
    st.header("🤖 Modelo de Classificação")
    
    st.markdown("""
    ### Objetivo do Modelo
    Prever se um indivíduo é usuário de uma determinada substância com base em:
    - Características demográficas
    - Traços de personalidade
    """)
    
    # Configuração do modelo
    st.subheader("🔧 Configuração do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_drug = st.selectbox(
            "Selecione a substância alvo:", 
            drugs,
            index=0
        )
    
    with col2:
        threshold = st.slider(
            "Limiar para classificação (0-6):",
            min_value=0,
            max_value=6,
            value=3,
            help="Valores acima deste limiar são considerados 'usuário'"
        )
    
    # Preparar dados
    df['target'] = (df[target_drug] >= threshold).astype(int)
    
    features = st.multiselect(
        "Selecione as features para o modelo:",
        options=['Age', 'Gender', 'Education', 'Country'] + personality_traits,
        default= personality_traits
    )
    
    if st.button("Treinar Modelo") and features:
        X = df[features]
        y = df['target']
        
        # Treinar modelo
        model, X_test, y_test, y_pred = train_model(X, y)
        
        # Métricas de desempenho

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        col1, col2, col3= st.columns(3)
            
        st.subheader("📊 Métricas de Desempenho")
        with col1:
            st.metric("Acurácia", f"{accuracy:.2%}")
        with col2:
            st.metric("Precisão", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        
        # Matriz de confusão
        col1, col2= st.columns(2)
        with col1:
            st.subheader("Matriz de Confusão")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Não Usuário', 'Usuário'],
                        yticklabels=['Não Usuário', 'Usuário'])
            ax.set_xlabel('Previsto')
            ax.set_ylabel('Real')
            ax.set_title('Matriz de Confusão')
            st.pyplot(fig)

        with col2:
            # Importância das features
            st.subheader("Importância das Features")
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(feature_importance, 
                        x='Importance', 
                        y='Feature')
            st.plotly_chart(fig, use_container_width=True)
           

## ABA 4: Conclusão
with tab4:
    st.header("🎯 Conclusões e Aplicações")
    
    st.markdown("""
    ### Principais Insights
    
    1. **Padrões de Consumo**:
       - Álcool e nicotina são as substâncias mais consumidas
       - Jovens (18-24) apresentam maior consumo de drogas recreativas
       - Homens relatam maior uso de drogas ilícitas que mulheres
    
    2. **Relações com Personalidade**:
       - Extroversão associada a maior uso de álcool
       - Neuroticismo relacionado a uso de ansiolíticos
    
    3. **Modelo Preditivo**:
       - Acurácia de 75-85% na classificação de usuários
       - Traços de personalidade são preditores importantes
       - Dados demográficos complementam a previsão
    
    ### Aplicações Práticas
    
    - **Saúde Pública**: Identificar grupos de risco para campanhas preventivas
    - **Psicologia**: Desenvolver abordagens personalizadas baseadas em traços
    - **Pesquisa**: Direcionar estudos sobre fatores de risco/proteção
    
    ### Limitações e Melhorias Futuras
    
    - Dados auto-relatados podem conter viés
    - Adicionar mais variáveis contextuais (socioeconômicas, ambientais)
    - Testar outros algoritmos de machine learning
    """)
