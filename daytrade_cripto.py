import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from tradingview_ta import Analysis, Interval
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Mammon Trade - Criptoativos",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💰 Mammon Trade - Análise de Criptoativos")
st.markdown("---")

# ---- Cache de dados ----
@st.cache_data(ttl=300)
def baixar_dados_tradingview(simbolo, periodo_dias=30):
    """Baixar dados do YFinance (dados que o TradingView usa)"""
    try:
        data_inicio = datetime.now() - timedelta(days=periodo_dias)
        df = yf.download(simbolo, start=data_inicio, end=datetime.now(), progress=False)
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return None

def baixar_dados_yfinance(simbolo, periodo_dias=30):
    """Baixar dados do YFinance"""
    try:
        data_inicio = datetime.now() - timedelta(days=periodo_dias)
        df = yf.download(simbolo, start=data_inicio, end=datetime.now(), progress=False)
        return df
    except Exception as e:
        st.error(f"Erro ao baixar dados: {e}")
        return None

def obter_analise_tradingview(simbolo):
    """Obter análise técnica do TradingView"""
    try:
        # Remover -USD e ajustar símbolo
        cripto = simbolo.replace('-USD', '')
        
        # Mapear para símbolos do TradingView
        mapa_tv = {
            'BTC': 'BTCUSD',
            'ETH': 'ETHUSD',
            'ADA': 'ADAUSD',
            'XRP': 'XRPUSD',
            'SOL': 'SOLUSD',
            'DOT': 'DOTUSD',
            'DOGE': 'DOGEUSD',
            'AXS': 'AXSUSD',
            'PEPE': 'PEPEUSD',
            'AAVE': 'AAVEUSD',
            'RENDER': 'RENDERUSD',
            'AVAX': 'AVAXUSD',
            'SAND': 'SANDUSD',
            'MANA': 'MANAUSD',
            'BIO': 'BIOUSD',
            'BERA': 'BERAUSD',
            'TRX': 'TRXUSD',
            'APE': 'APEUSD',
            'SUSHI': 'SUSHIUSD',
            'LDO': 'LDOUSD',
            'LTC': 'LTCUSD',
            'CHZ': 'CHZUSD',
            'WIF': 'WIFUSD',
            'NEIRO': 'NEIROUSD',
            'SHIB': 'SHIBUSD'
        }
        
        tv_symbol = mapa_tv.get(cripto, cripto + 'USD')
        
        # Fazer análise em diferentes timeframes
        try:
            analise_1h = Analysis(
                screener="crypto",
                interval=Interval.INTERVAL_1H,
                symbol=tv_symbol
            )
        except:
            analise_1h = None
        
        try:
            analise_4h = Analysis(
                screener="crypto",
                interval=Interval.INTERVAL_4H,
                symbol=tv_symbol
            )
        except:
            analise_4h = None
        
        try:
            analise_1d = Analysis(
                screener="crypto",
                interval=Interval.INTERVAL_1D,
                symbol=tv_symbol
            )
        except:
            analise_1d = None
        
        return {
            '1h': analise_1h,
            '4h': analise_4h,
            '1d': analise_1d
        }
    except Exception as e:
        st.warning(f"Análise TradingView indisponível: {e}")
        return None

def calcular_indicadores(df):
    """Calcular todos os indicadores técnicos de forma robusta"""
    try:
        df = df.copy()
        
        if df is None or df.empty or len(df) < 2:
            return None
        
        # Garantir que Close é um Series unidimensional
        close_prices = df['Close'].squeeze()
        if isinstance(close_prices, np.ndarray):
            close_prices = pd.Series(close_prices, index=df.index)
        
        # Cálculos manuais dos indicadores
        
        # RSI - Índice de Força Relativa
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # Bandas de Bollinger
        sma = close_prices.rolling(window=20).mean()
        std = close_prices.rolling(window=20).std()
        df['BB_Upper'] = sma + (std * 2)
        df['BB_Middle'] = sma
        df['BB_Lower'] = sma - (std * 2)
        
        # EMA
        df['EMA_12'] = close_prices.ewm(span=12).mean()
        df['EMA_26'] = close_prices.ewm(span=26).mean()
        
        # Retorno Diário
        df['Retorno'] = close_prices.pct_change() * 100
        
        # Volatilidade
        df['Volatilidade'] = df['Retorno'].rolling(window=20).std()
        
        # Preencher valores NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Erro ao calcular indicadores: {e}")
        return None

def gerar_sinais(df):
    """Gerar sinais de compra e venda com regras bem definidas"""
    df = df.copy()
    df['Sinal'] = ''
    df['Força_Sinal'] = 0
    
    for i in range(1, len(df)):
        sinais = []
        forca = 0
        
        try:
            # Extrair valores escalares
            close_atual = float(df['Close'].iloc[i])
            close_anterior = float(df['Close'].iloc[i-1])
            rsi_atual = float(df['RSI'].iloc[i]) if 'RSI' in df.columns else 50
            rsi_anterior = float(df['RSI'].iloc[i-1]) if 'RSI' in df.columns else 50
            
            macd_atual = float(df['MACD'].iloc[i]) if 'MACD' in df.columns else 0
            macd_signal_atual = float(df['MACD_Signal'].iloc[i]) if 'MACD_Signal' in df.columns else 0
            macd_anterior = float(df['MACD'].iloc[i-1]) if 'MACD' in df.columns else 0
            macd_signal_anterior = float(df['MACD_Signal'].iloc[i-1]) if 'MACD_Signal' in df.columns else 0
            
            bb_upper = float(df['BB_Upper'].iloc[i]) if 'BB_Upper' in df.columns else close_atual * 1.05
            bb_lower = float(df['BB_Lower'].iloc[i]) if 'BB_Lower' in df.columns else close_atual * 0.95
            
            ema_12 = float(df['EMA_12'].iloc[i]) if 'EMA_12' in df.columns else close_atual
            ema_26 = float(df['EMA_26'].iloc[i]) if 'EMA_26' in df.columns else close_atual
            ema_12_ant = float(df['EMA_12'].iloc[i-1]) if 'EMA_12' in df.columns else close_anterior
            ema_26_ant = float(df['EMA_26'].iloc[i-1]) if 'EMA_26' in df.columns else close_anterior
            
            # SINAL DE COMPRA
            if rsi_atual < 30:  # RSI sobrevendido
                sinais.append("RSI Baixo")
                forca += 2
                
            if close_atual < bb_lower:  # Preço abaixo da Banda Inferior
                sinais.append("BB Inferior")
                forca += 2
                
            if macd_atual > macd_signal_atual and macd_anterior <= macd_signal_anterior:  # Cruzamento MACD
                sinais.append("MACD Crossover")
                forca += 2
                
            if ema_12 > ema_26 and ema_12_ant <= ema_26_ant:  # EMA Crossover
                sinais.append("EMA Crossover")
                forca += 1
            
            # SINAL DE VENDA
            if rsi_atual > 70:  # RSI sobrecomprado
                sinais.append("RSI Alto")
                forca -= 2
                
            if close_atual > bb_upper:  # Preço acima da Banda Superior
                sinais.append("BB Superior")
                forca -= 2
                
            if macd_atual < macd_signal_atual and macd_anterior >= macd_signal_anterior:  # Cruzamento MACD
                sinais.append("MACD Crossover Down")
                forca -= 2
            
            if forca >= 3:
                df.loc[df.index[i], 'Sinal'] = '🟢 COMPRA'
            elif forca <= -3:
                df.loc[df.index[i], 'Sinal'] = '🔴 VENDA'
            
            df.loc[df.index[i], 'Força_Sinal'] = forca
        except Exception as e:
            # Se houver erro em uma linha, continua para a próxima
            continue
    
    return df

def calcular_estatisticas(df):
    """Calcular estatísticas de performance"""
    stats = {}
    
    retornos = df['Retorno'].dropna()
    stats['Retorno Médio'] = float(retornos.mean()) if len(retornos) > 0 else 0.0
    stats['Volatilidade Média'] = float(df['Volatilidade'].mean()) if len(df) > 0 else 0.0
    stats['Retorno Total'] = float(((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100) if len(df) > 0 else 0.0
    stats['Max Diário'] = float(retornos.max()) if len(retornos) > 0 else 0.0
    stats['Min Diário'] = float(retornos.min()) if len(retornos) > 0 else 0.0
    stats['Sharpe Ratio'] = float((retornos.mean() / retornos.std()) * np.sqrt(252)) if (retornos.std() > 0 and len(retornos) > 0) else 0.0
    
    return stats

# ---- Sidebar ----
st.sidebar.header("⚙️ Configurações")

criptos = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Cardano': 'ADA-USD',
    'Ripple': 'XRP-USD',
    'Solana': 'SOL-USD',
    'Polkadot': 'DOT-USD',
    'Dogecoin': 'DOGE-USD',
    'Axie Infinity': 'AXS-USD',
    'Pepe': 'PEPE-USD',
    'Aave': 'AAVE-USD',
    'Render': 'RENDER-USD',
    'Avalanche': 'AVAX-USD',
    'Sandbox': 'SAND-USD',
    'Decentraland': 'MANA-USD',
    'Biop': 'BIO-USD',
    'Berachain': 'BERA-USD',
    'Tron': 'TRX-USD',
    'ApeCoin': 'APE-USD',
    'Sushi': 'SUSHI-USD',
    'Lido': 'LDO-USD',
    'Litecoin': 'LTC-USD',
    'Chiliz': 'CHZ-USD',
    'WIF': 'WIF-USD',
    'Neiro': 'NEIRO-USD',
    'Shiba Inu': 'SHIB-USD'
}

cripto_selecionada = st.sidebar.selectbox(
    "Selecione a Criptomoeda",
    list(criptos.keys()),
    index=0
)

simbolo = criptos[cripto_selecionada]

intervalo = st.sidebar.radio(
    "Intervalo Temporal",
    ["1h", "4h", "1d", "1w", "1m", "1y"],
    index=2
)

# Mapear intervalo para período em dias
mapa_periodo = {
    "1h": 7,
    "4h": 7,
    "1d": 30,
    "1w": 90,
    "1m": 180,
    "1y": 365
}

periodo_dias = mapa_periodo.get(intervalo, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Indicadores")
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=True)
show_bb = st.sidebar.checkbox("Bandas de Bollinger", value=True)
show_ema = st.sidebar.checkbox("EMA", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Gerenciamento de Risco")
capital_inicial = st.sidebar.number_input(
    "Capital Inicial (R$)",
    min_value=100.0,
    value=1000.0,
    step=100.0
)

risco_por_trade = st.sidebar.slider(
    "Risco por Trade (%)",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5
)

take_profit = st.sidebar.slider(
    "Take Profit (%)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5
)

stop_loss = st.sidebar.slider(
    "Stop Loss (%)",
    min_value=0.5,
    max_value=10.0,
    value=1.0,
    step=0.5
)

# ---- Carregar e processar dados ----
st.subheader(f"📈 {cripto_selecionada}")

with st.spinner("Carregando dados do TradingView..."):
    # Carregar dados (usando YFinance, mesmos dados que TradingView analisa)
    df = baixar_dados_tradingview(simbolo, periodo_dias)
    
    if df is not None and not df.empty:
        df = calcular_indicadores(df)
        df = gerar_sinais(df)
        
        # Obter análise do TradingView
        analise_tv = obter_analise_tradingview(simbolo)
        
        # Mostrar análise TradingView se disponível
        if analise_tv:
            st.markdown("---")
            st.markdown("### 🎯 Análise Técnica do TradingView em Tempo Real")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### ⏱️ 1 Hora")
                try:
                    if analise_tv['1h']:
                        analise_dict = analise_tv['1h'].get_analysis()
                        recomendacao = analise_dict['summary']['RECOMMENDATION']
                        buy = analise_dict['summary']['BUY']
                        sell = analise_dict['summary']['SELL']
                        neutral = analise_dict['summary']['NEUTRAL']
                        
                        if recomendacao == 'BUY':
                            st.success(f"🟢 {recomendacao}")
                        elif recomendacao == 'SELL':
                            st.error(f"🔴 {recomendacao}")
                        else:
                            st.warning(f"🟡 {recomendacao}")
                        
                        st.write(f"Compra: {buy} | Venda: {sell} | Neutro: {neutral}")
                except:
                    st.write("📊 Dados indisponíveis")
            
            with col2:
                st.markdown("#### ⏱️ 4 Horas")
                try:
                    if analise_tv['4h']:
                        analise_dict = analise_tv['4h'].get_analysis()
                        recomendacao = analise_dict['summary']['RECOMMENDATION']
                        buy = analise_dict['summary']['BUY']
                        sell = analise_dict['summary']['SELL']
                        neutral = analise_dict['summary']['NEUTRAL']
                        
                        if recomendacao == 'BUY':
                            st.success(f"🟢 {recomendacao}")
                        elif recomendacao == 'SELL':
                            st.error(f"🔴 {recomendacao}")
                        else:
                            st.warning(f"🟡 {recomendacao}")
                        
                        st.write(f"Compra: {buy} | Venda: {sell} | Neutro: {neutral}")
                except:
                    st.write("📊 Dados indisponíveis")
            
            with col3:
                st.markdown("#### 📅 1 Dia")
                try:
                    if analise_tv['1d']:
                        analise_dict = analise_tv['1d'].get_analysis()
                        recomendacao = analise_dict['summary']['RECOMMENDATION']
                        buy = analise_dict['summary']['BUY']
                        sell = analise_dict['summary']['SELL']
                        neutral = analise_dict['summary']['NEUTRAL']
                        
                        if recomendacao == 'BUY':
                            st.success(f"🟢 {recomendacao}")
                        elif recomendacao == 'SELL':
                            st.error(f"🔴 {recomendacao}")
                        else:
                            st.warning(f"🟡 {recomendacao}")
                        
                        st.write(f"Compra: {buy} | Venda: {sell} | Neutro: {neutral}")
                except:
                    st.write("📊 Dados indisponíveis")
            
            st.markdown("---")
        
        # ---- Tabs principais ----
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏆 Melhores Ativos", "📊 Gráfico Técnico", "📋 Sinais", "📈 Estatísticas", "⚙️ Backtest", "🎯 TradingView"])
        
        with tab1:
            st.markdown("### 🏆 Ranking de Melhores Ativos do Dia")
            
            @st.cache_data(ttl=600)
            def analisar_todos_ativos():
                """Analisa todos os ativos e retorna ranking"""
                resultados = []
                
                for nome, simbolo_ativo in criptos.items():
                    try:
                        # Baixar dados
                        df_ativo = baixar_dados_tradingview(simbolo_ativo, 30)
                        if df_ativo is None or df_ativo.empty:
                            continue
                        
                        # Calcular indicadores
                        df_ativo = calcular_indicadores(df_ativo)
                        df_ativo = gerar_sinais(df_ativo)
                        
                        # Pegar último sinal e força
                        ultimo_sinal = df_ativo[df_ativo['Sinal'] != ''].tail(1)
                        
                        if not ultimo_sinal.empty:
                            forca = float(ultimo_sinal['Força_Sinal'].values[0])
                            sinal = ultimo_sinal['Sinal'].values[0]
                            preco = float(df_ativo['Close'].iloc[-1])
                            
                            # Calcular retorno do dia
                            retorno_dia = ((preco / float(df_ativo['Close'].iloc[0])) - 1) * 100
                            
                            resultados.append({
                                'Ativo': nome,
                                'Sinal': sinal,
                                'Força': abs(forca),
                                'Preço': preco,
                                'Retorno 30d': retorno_dia,
                                'Tipo': '🟢 COMPRA' if '🟢' in sinal else '🔴 VENDA'
                            })
                    except:
                        continue
                
                return pd.DataFrame(resultados) if resultados else pd.DataFrame()
            
            with st.spinner("⏳ Analisando todos os ativos... Isso pode levar alguns segundos"):
                df_ranking = analisar_todos_ativos()
            
            if not df_ranking.empty:
                # Separar compras e vendas
                compras = df_ranking[df_ranking['Tipo'] == '🟢 COMPRA'].sort_values('Força', ascending=False)
                vendas = df_ranking[df_ranking['Tipo'] == '🔴 VENDA'].sort_values('Força', ascending=False)
                
                col_compra, col_venda = st.columns(2)
                
                with col_compra:
                    st.markdown("#### 🟢 TOP 5 COMPRA")
                    if not compras.empty:
                        top_compra = compras.head(5)[['Ativo', 'Sinal', 'Força', 'Preço', 'Retorno 30d']].copy()
                        top_compra['Preço'] = top_compra['Preço'].apply(lambda x: f"${x:.2f}")
                        top_compra['Retorno 30d'] = top_compra['Retorno 30d'].apply(lambda x: f"{x:.2f}%")
                        top_compra['Força'] = top_compra['Força'].apply(lambda x: f"{int(x)}")
                        
                        st.dataframe(
                            top_compra.reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Melhores oportunidades de compra com destaque
                        for idx, row in compras.head(3).iterrows():
                            st.success(f"**{idx + 1}. {row['Ativo']}** - Força: {int(row['Força'])} ⭐")
                    else:
                        st.info("Nenhum sinal de compra identificado")
                
                with col_venda:
                    st.markdown("#### 🔴 TOP 5 VENDA")
                    if not vendas.empty:
                        top_venda = vendas.head(5)[['Ativo', 'Sinal', 'Força', 'Preço', 'Retorno 30d']].copy()
                        top_venda['Preço'] = top_venda['Preço'].apply(lambda x: f"${x:.2f}")
                        top_venda['Retorno 30d'] = top_venda['Retorno 30d'].apply(lambda x: f"{x:.2f}%")
                        top_venda['Força'] = top_venda['Força'].apply(lambda x: f"{int(x)}")
                        
                        st.dataframe(
                            top_venda.reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Melhores oportunidades de venda com destaque
                        for idx, row in vendas.head(3).iterrows():
                            st.error(f"**{idx + 1}. {row['Ativo']}** - Força: {int(row['Força'])} ⚠️")
                    else:
                        st.info("Nenhum sinal de venda identificado")
                
                st.markdown("---")
                st.markdown("#### 📊 Resumo Geral")
                col_resumo1, col_resumo2, col_resumo3 = st.columns(3)
                
                with col_resumo1:
                    st.metric("Total de Ativos", len(df_ranking))
                with col_resumo2:
                    st.metric("🟢 Sinais de Compra", len(compras))
                with col_resumo3:
                    st.metric("🔴 Sinais de Venda", len(vendas))
            else:
                st.warning("Aguardando análise dos ativos...")
        
        with tab2:
            st.markdown("### Análise Técnica")
            
            # Gráfico Principal
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Preço',
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Bandas de Bollinger
            if show_bb:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['BB_Upper'],
                    name='BB Superior',
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['BB_Lower'],
                    name='BB Inferior',
                    mode='lines',
                    line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(100, 100, 100, 0.1)',
                    showlegend=True
                ))
            
            # EMA
            if show_ema:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA_12'],
                    name='EMA 12',
                    mode='lines',
                    line=dict(color='blue', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['EMA_26'],
                    name='EMA 26',
                    mode='lines',
                    line=dict(color='orange', width=1)
                ))
            
            fig.update_layout(
                title=f'{cripto_selecionada} - Análise Técnica',
                yaxis_title='Preço (USD)',
                template='plotly_dark',
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI
            if show_rsi:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=df.index, y=df['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido")
                fig_rsi.update_layout(
                    title='Índice de Força Relativa (RSI)',
                    yaxis_title='RSI',
                    template='plotly_dark',
                    height=300,
                    hovermode='x'
                )
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            # MACD
            if show_macd:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=df.index, y=df['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=df.index, y=df['MACD_Signal'],
                    name='Signal',
                    line=dict(color='red')
                ))
                fig_macd.add_trace(go.Bar(
                    x=df.index, y=df['MACD_Diff'],
                    name='Histograma',
                    marker_color='rgba(128, 128, 128, 0.3)'
                ))
                fig_macd.update_layout(
                    title='MACD - Moving Average Convergence Divergence',
                    yaxis_title='MACD',
                    template='plotly_dark',
                    height=300,
                    hovermode='x'
                )
                st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            st.markdown("### 🎯 Sinais de Operação")
            
            sinais_df = df[df['Sinal'] != ''][['Close', 'Sinal', 'RSI', 'MACD', 'Força_Sinal']].tail(20).copy()
            sinais_df.index.name = 'Data'
            
            if not sinais_df.empty:
                st.dataframe(
                    sinais_df,
                    use_container_width=True
                )
                
                # Resumo dos sinais
                compras = len(df[df['Sinal'] == '🟢 COMPRA'])
                vendas = len(df[df['Sinal'] == '🔴 VENDA'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🟢 Sinais de Compra", compras)
                with col2:
                    st.metric("🔴 Sinais de Venda", vendas)
                
                # Último sinal
                ultimo_sinal = df[df['Sinal'] != ''].tail(1)
                if not ultimo_sinal.empty:
                    st.success(f"**Último Sinal:** {ultimo_sinal['Sinal'].values[0]} em {ultimo_sinal.index[0].strftime('%d/%m/%Y %H:%M')}")
                    st.info(f"**Força do Sinal:** {abs(int(ultimo_sinal['Força_Sinal'].values[0]))} (quanto maior, mais confiável)")
            else:
                st.warning("Nenhum sinal gerado para este período")
        
        with tab4:
            st.markdown("### 📊 Estatísticas de Performance")
            
            stats = calcular_estatisticas(df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Retorno Total",
                    f"{stats['Retorno Total']:.2f}%",
                    delta=f"{stats['Retorno Médio']:.2f}% ao dia"
                )
            with col2:
                st.metric(
                    "Volatilidade Média",
                    f"{stats['Volatilidade Média']:.2f}%"
                )
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{stats['Sharpe Ratio']:.2f}"
                )
            
            col4, col5 = st.columns(2)
            with col4:
                st.metric("Melhor Dia", f"{stats['Max Diário']:.2f}%")
            with col5:
                st.metric("Pior Dia", f"{stats['Min Diário']:.2f}%")
            
            # Gráfico de retorno acumulado
            df_retorno = df.copy()
            df_retorno['Retorno_Acumulado'] = (1 + df_retorno['Retorno'] / 100).cumprod()
            
            # Resetar índice para plotly funcionar
            df_plot = df_retorno[['Retorno_Acumulado']].reset_index()
            if 'Ticker' in df_plot.columns:
                df_plot = df_plot.drop('Ticker', axis=1)
            
            fig_retorno = go.Figure()
            fig_retorno.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot['Retorno_Acumulado'],
                name='Retorno Acumulado',
                line=dict(color='green', width=2),
                mode='lines'
            ))
            fig_retorno.update_layout(
                title='Retorno Acumulado',
                yaxis_title='Retorno Acumulado',
                xaxis_title='Data',
                template='plotly_dark',
                height=400,
                hovermode='x'
            )
            st.plotly_chart(fig_retorno, use_container_width=True)
        
        with tab5:
            st.markdown("### 🔄 Simulação de Backtest")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Capital Inicial", f"R$ {capital_inicial:.2f}")
            with col2:
                st.metric("Risco/Trade", f"{risco_por_trade}%")
            with col3:
                st.metric("Take Profit", f"{take_profit}%")
            
            st.markdown("---")
            
            # Simular trades
            capital = capital_inicial
            trades = []
            
            for i in range(1, len(df)):
                if df['Sinal'].iloc[i] == '🟢 COMPRA':
                    entrada = float(df['Close'].iloc[i])
                    tp = entrada * (1 + take_profit / 100)
                    sl = entrada * (1 - stop_loss / 100)
                    
                    # Simular fechamento do trade
                    for j in range(i+1, min(i+20, len(df))):
                        close_j = float(df['Close'].iloc[j])
                        if close_j >= tp:
                            lucro = capital * (risco_por_trade / 100)
                            capital += lucro
                            trades.append({
                                'Data Entrada': df.index[i],
                                'Entrada': entrada,
                                'Saída': tp,
                                'Tipo': 'TP',
                                'Lucro': lucro,
                                'Capital': capital
                            })
                            break
                        elif close_j <= sl:
                            perda = capital * (risco_por_trade / 100)
                            capital -= perda
                            trades.append({
                                'Data Entrada': df.index[i],
                                'Entrada': entrada,
                                'Saída': sl,
                                'Tipo': 'SL',
                                'Lucro': -perda,
                                'Capital': capital
                            })
                            break
            
            if trades:
                trades_df = pd.DataFrame(trades)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Capital Final",
                        f"R$ {capital:.2f}",
                        delta=f"R$ {capital - capital_inicial:.2f}"
                    )
                with col2:
                    taxa_acerto = (len(trades_df[trades_df['Tipo'] == 'TP']) / len(trades_df)) * 100
                    st.metric("Taxa de Acerto", f"{taxa_acerto:.1f}%")
                with col3:
                    total_trades = len(trades_df)
                    st.metric("Total de Trades", total_trades)
                
                st.markdown("---")
                st.dataframe(trades_df, use_container_width=True)
                
                # Gráfico de evolução de capital
                fig_capital = px.line(
                    trades_df,
                    x='Data Entrada',
                    y='Capital',
                    title='Evolução do Capital',
                    labels={'Capital': 'Capital (R$)'},
                    template='plotly_dark'
                )
                fig_capital.update_layout(height=400)
                st.plotly_chart(fig_capital, use_container_width=True)
            else:
                st.warning("Nenhum trade executado no período")
        
        with tab6:
            st.markdown("### 📊 Gráfico do TradingView em Tempo Real")
            
            # Mapear símbolo para o TradingView
            cripto_tv = simbolo.replace('-USD', '')
            mapa_tv_symbols = {
                'BTC': 'BTCUSD',
                'ETH': 'ETHUSD',
                'ADA': 'ADAUSD',
                'XRP': 'XRPUSD',
                'SOL': 'SOLUSD',
                'DOT': 'DOTUSD',
                'DOGE': 'DOGEUSD',
                'AXS': 'AXSUSD',
                'PEPE': 'PEPEUSD',
                'AAVE': 'AAVEUSD',
                'RENDER': 'RENDERUSD',
                'AVAX': 'AVAXUSD',
                'SAND': 'SANDUSD',
                'MANA': 'MANAUSD',
                'BIO': 'BIOUSD',
                'BERA': 'BERAUSD',
                'TRX': 'TRXUSD',
                'APE': 'APEUSD',
                'SUSHI': 'SUSHIUSD',
                'LDO': 'LDOUSD',
                'LTC': 'LTCUSD',
                'CHZ': 'CHZUSD',
                'WIF': 'WIFUSD',
                'NEIRO': 'NEIROUSD',
                'SHIB': 'SHIBUSD'
            }
            
            tv_symbol = mapa_tv_symbols.get(cripto_tv, cripto_tv + 'USD')
            
            # HTML para o widget do TradingView
            tradingview_html = f"""
            <!-- TradingView Widget BEGIN -->
            <div class="tradingview-widget-container">
              <div id="tradingview_chart"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget(
              {{
              "autosize": true,
              "symbol": "CRYPTO:{tv_symbol}",
              "interval": "D",
              "timezone": "exchange",
              "theme": "dark",
              "style": "1",
              "locale": "br",
              "toolbar_bg": "#f1f3f6",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "container_id": "tradingview_chart"
            }}
              );
              </script>
            </div>
            <!-- TradingView Widget END -->
            <style>
              .tradingview-widget-container {{
                width: 100%;
                height: 600px;
              }}
            </style>
            """
            
            st.components.v1.html(tradingview_html, height=900, scrolling=False)
            
            st.markdown("---")
            st.info("💡 **Dica:** Você pode mudar o intervalo (D, W, M), incluir indicadores e fazer análises direto no gráfico do TradingView!")
    else:
        st.error("Não foi possível carregar os dados")

        #para rodar, escreva no terminal: streamlit run daytrade_cripto.py
