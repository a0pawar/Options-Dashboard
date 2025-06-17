import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.signal import savgol_filter
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NSE Options Surveillance Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, minimalist CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #2E3440;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.4rem;
        color: #3B4252;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E9F0;
        padding-bottom: 0.5rem;
    }
    .interpretation-box {
        background-color: #F8F9FA;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #5E81AC;
        margin: 1rem 0;
    }
    .metric-highlight {
        background-color: #ECEFF4;
        padding: 0.8rem;
        border-radius: 6px;
        text-align: center;
        border: 1px solid #D8DEE9;
    }
    .warning-box {
        background-color: #FEF3E2;
        padding: 1rem;
        border-radius: 6px;
        border-left: 4px solid #F39C12;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">NSE Options Market Surveillance</h1>', unsafe_allow_html=True)

# Configuration
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:138.0) Gecko/20100101 Firefox/138.0",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "DNT": "1",
    "Sec-GPC": "1",
}

# Black-Scholes Functions
@st.cache_data
def black_scholes_price(S, K, T, r, q, sigma, option_type='CE'):
    """Calculates Black-Scholes option price."""
    if sigma <= 1e-6 or T <= 1e-6:
        if option_type == 'CE':
            return max(0, S * np.exp(-q * T) - K * np.exp(-r * T))
        elif option_type == 'PE':
            return max(0, K * np.exp(-r * T) - S * np.exp(-q * T))
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'PE':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'CE' or 'PE'")
    return price

@st.cache_data
def black_scholes_vega(S, K, T, r, q, sigma):
    """Calculates Vega of an option."""
    if sigma <= 1e-6 or T <= 1e-6: 
        return 1e-9
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    return vega

# Data fetching functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_option_chain_data(symbol, max_expiries=4):
    """Fetch option chain data from NSE"""
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    
    # Warm-up request
    warmup_url = "https://www.nseindia.com/option-chain"
    try:
        page_load_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        response_warmup = session.get(warmup_url, headers=page_load_headers, timeout=15)
        response_warmup.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection failed: {e}")
        return None
    
    # Fetch expiry dates
    contract_info_url = f"https://www.nseindia.com/api/option-chain-contract-info?symbol={symbol}"
    try:
        contract_api_headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "X-Requested-With": "XMLHttpRequest",
            "Referer": warmup_url,
        }
        response_contracts = session.get(contract_info_url, headers=contract_api_headers, timeout=15)
        response_contracts.raise_for_status()
        contract_data = response_contracts.json()
        
        if 'expiryDates' not in contract_data:
            st.error("No expiry dates found")
            return None
            
        expiry_dates = contract_data['expiryDates'][:max_expiries]
        
    except Exception as e:
        st.error(f"Failed to fetch contract info: {e}")
        return None
    
    # Fetch option chain data for each expiry
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, expiry_date in enumerate(expiry_dates):
        status_text.text(f"Fetching data for expiry: {expiry_date}")
        
        api_url = f"https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol={symbol}&expiry={expiry_date}"
        
        try:
            api_headers = {
                "Accept": "application/json, text/javascript, */*; q=0.01",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": warmup_url,
            }
            
            response_api = session.get(api_url, headers=api_headers, timeout=15)
            response_api.raise_for_status()
            data = response_api.json()
            
            # Process data
            if 'records' in data and 'data' in data['records']:
                raw_option_data = data['records']['data']
            elif 'filtered' in data and 'data' in data['filtered']:
                raw_option_data = data['filtered']['data']
            elif 'data' in data:
                raw_option_data = data['data']
            else:
                continue
                
            for record in raw_option_data:
                flat_record = {}
                flat_record['strikePrice'] = record.get('strikePrice')
                flat_record['record_expiryDate'] = record.get('expiryDate')
                flat_record['api_call_expiry_date'] = expiry_date
                
                # CE data
                ce_data = record.get('CE', {})
                if ce_data:
                    for key, value in ce_data.items():
                        flat_record[f'CE_{key}'] = value
                        
                # PE data
                pe_data = record.get('PE', {})
                if pe_data:
                    for key, value in pe_data.items():
                        flat_record[f'PE_{key}'] = value
                        
                all_data.append(flat_record)
                
        except Exception as e:
            st.warning(f"Failed to fetch data for expiry {expiry_date}: {e}")
            
        progress_bar.progress((i + 1) / len(expiry_dates))
    
    status_text.empty()
    progress_bar.empty()
    
    return pd.DataFrame(all_data) if all_data else None

def process_option_data(df_raw, risk_free_rate, dividend_yield):
    """Process raw option chain data"""
    if df_raw is None or df_raw.empty:
        return None, None, None, None
        
    # Clean and structure data
    ce_data_list = []
    pe_data_list = []
    
    for index, row in df_raw.iterrows():
        strike_price = row['strikePrice']
        try:
            expiry_date = pd.to_datetime(row['api_call_expiry_date'], format='%d-%b-%Y')
        except ValueError:
            expiry_date = pd.NaT
            
        # CE Data
        if pd.notna(row.get('CE_expiryDate')):
            ce_data_list.append({
                'expiryDate': expiry_date,
                'underlying': row.get('CE_underlying'),
                'underlyingValue': row.get('CE_underlyingValue'),
                'strikePrice': strike_price,
                'optionType': 'CE',
                'openInterest': row.get('CE_openInterest'),
                'changeinOpenInterest': row.get('CE_changeinOpenInterest'),
                'totalTradedVolume': row.get('CE_totalTradedVolume'),
                'impliedVolatility': row.get('CE_impliedVolatility'),
                'lastPrice': row.get('CE_lastPrice'),
                'change': row.get('CE_change'),
                'totalBuyQuantity': row.get('CE_totalBuyQuantity'),
                'totalSellQuantity': row.get('CE_totalSellQuantity')
            })
            
        # PE Data
        if pd.notna(row.get('PE_expiryDate')):
            pe_data_list.append({
                'expiryDate': expiry_date,
                'underlying': row.get('PE_underlying'),
                'underlyingValue': row.get('PE_underlyingValue'),
                'strikePrice': strike_price,
                'optionType': 'PE',
                'openInterest': row.get('PE_openInterest'),
                'changeinOpenInterest': row.get('PE_changeinOpenInterest'),
                'totalTradedVolume': row.get('PE_totalTradedVolume'),
                'impliedVolatility': row.get('PE_impliedVolatility'),
                'lastPrice': row.get('PE_lastPrice'),
                'change': row.get('PE_change'),
                'totalBuyQuantity': row.get('PE_totalBuyQuantity'),
                'totalSellQuantity': row.get('PE_totalSellQuantity')
            })
    
    df_ce = pd.DataFrame(ce_data_list)
    df_pe = pd.DataFrame(pe_data_list)
    df_options = pd.concat([df_ce, df_pe], ignore_index=True)
    
    if df_options.empty:
        return None, None, None, None
    
    # Convert numeric columns
    numeric_cols = ['underlyingValue', 'strikePrice', 'openInterest', 'changeinOpenInterest',
                   'totalTradedVolume', 'impliedVolatility', 'lastPrice', 'change',
                   'totalBuyQuantity', 'totalSellQuantity']
    for col in numeric_cols:
        df_options[col] = pd.to_numeric(df_options[col], errors='coerce')
    
    df_options['expiryDate'] = pd.to_datetime(df_options['expiryDate'])
    df_options = df_options[df_options['underlyingValue'].notna() & (df_options['underlyingValue'] > 0)]
    df_options = df_options[df_options['expiryDate'].notna()]
    
    # Calculate spot prices
    spot_prices_map = df_options.groupby('expiryDate')['underlyingValue'].mean().to_dict()
    df_options['currentSpotPrice'] = df_options['expiryDate'].map(spot_prices_map)
    df_options = df_options[df_options['currentSpotPrice'].notna()]
    
    # Calculate time to expiry
    today = dt.datetime.now()
    df_options['timeToExpiry'] = (df_options['expiryDate'] - today).dt.days / 365.25
    
    if df_options.empty:
        return None, None, None, None
    
    # Calculate Put-Call Ratios
    pcr_data = []
    for expiry in df_options['expiryDate'].unique():
        expiry_data = df_options[df_options['expiryDate'] == expiry]
        
        total_oi_puts = expiry_data[expiry_data['optionType'] == 'PE']['openInterest'].sum()
        total_oi_calls = expiry_data[expiry_data['optionType'] == 'CE']['openInterest'].sum()
        total_vol_puts = expiry_data[expiry_data['optionType'] == 'PE']['totalTradedVolume'].sum()
        total_vol_calls = expiry_data[expiry_data['optionType'] == 'CE']['totalTradedVolume'].sum()
        
        pcr_oi = total_oi_puts / total_oi_calls if total_oi_calls > 0 else np.nan
        pcr_vol = total_vol_puts / total_vol_calls if total_vol_calls > 0 else np.nan
        
        pcr_data.append({
            'expiryDate': expiry,
            'PCR_OpenInterest': pcr_oi,
            'PCR_Volume': pcr_vol,
            'Total_OI_Puts': total_oi_puts,
            'Total_OI_Calls': total_oi_calls,
            'Total_Vol_Puts': total_vol_puts,
            'Total_Vol_Calls': total_vol_calls,
            'daysToExpiry': (expiry - today).days
        })
    
    df_pcr = pd.DataFrame(pcr_data)
    
    # Calculate crash probabilities for puts (1st and 4th expiry only)
    df_puts = df_options[df_options['optionType'] == 'PE'].copy()
    df_puts = df_puts[df_puts['impliedVolatility'] > 1e-6]
    
    # Calculate delta for all puts first
    def calculate_d1_bs(S, K, T, r, q, sigma_pct):
        sigma = sigma_pct / 100.0
        if sigma <= 1e-6 or T <= 1e-6: 
            return np.nan
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def calculate_put_delta_bs(d1, T):
        if pd.isna(d1) or T <= 1e-6: 
            return np.nan
        return norm.cdf(d1) - 1
    
    if not df_puts.empty:
        df_puts['d1'] = df_puts.apply(
            lambda row: calculate_d1_bs(
                S=row['currentSpotPrice'], K=row['strikePrice'], T=row['timeToExpiry'],
                r=risk_free_rate, q=dividend_yield, sigma_pct=row['impliedVolatility']
            ), axis=1
        )
        df_puts['delta'] = df_puts.apply(
            lambda row: calculate_put_delta_bs(d1=row['d1'], T=row['timeToExpiry']), axis=1
        )
        df_puts['prob_ST_less_K'] = np.abs(df_puts['delta'])
    
    crash_probabilities_data = []
    if not df_puts.empty:
        # Get 1st and 4th expiry for crash analysis
        sorted_expiries = sorted(df_puts['expiryDate'].unique())
        target_expiries = []
        if len(sorted_expiries) >= 1:
            target_expiries.append(sorted_expiries[0])  # 1st expiry
        if len(sorted_expiries) >= 4:
            target_expiries.append(sorted_expiries[3])  # 4th expiry
        
        # Calculate crash probabilities for target expiries only
        df_otm_puts = df_puts[
            (df_puts['strikePrice'] < df_puts['currentSpotPrice']) &
            (df_puts['expiryDate'].isin(target_expiries))
        ].copy()
        
        crash_thresholds_pct = [-0.05, -0.1, -0.15, -0.2]
        
        for expiry in target_expiries:
            expiry_puts = df_otm_puts[df_otm_puts['expiryDate'] == expiry]
            if expiry_puts.empty:
                continue
                
            current_spot = expiry_puts['currentSpotPrice'].iloc[0]
            
            for pct_drop in crash_thresholds_pct:
                target_strike = current_spot * (1 + pct_drop)
                closest_option = expiry_puts[expiry_puts['strikePrice'] <= target_strike].sort_values(
                    by='strikePrice', ascending=False).head(1)
                
                prob = np.nan
                actual_strike_found = np.nan
                if not closest_option.empty:
                    prob = closest_option['prob_ST_less_K'].iloc[0]
                    actual_strike_found = closest_option['strikePrice'].iloc[0]
                
                crash_probabilities_data.append({
                    'expiryDate': expiry,
                    'spotPrice': current_spot,
                    'crashThresholdPct': pct_drop,
                    'targetStrike': target_strike,
                    'closestStrikeFound': actual_strike_found,
                    'crashProbability': prob if pd.notna(prob) else 0.0
                })
    
    df_crash_probs = pd.DataFrame(crash_probabilities_data)
    
    return df_options, df_pcr, df_crash_probs, df_puts

# Sidebar
with st.sidebar:
    st.markdown('<h3 style="color: #3B4252;">⚙️ Configuration</h3>', unsafe_allow_html=True)
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Index",
        ["NIFTY", "BANKNIFTY", "FINNIFTY"],
        index=0
    )
    
    # Parameters
    st.markdown("**Analysis Parameters**")
    risk_free_rate = st.slider("Risk Free Rate (%)", 1.0, 10.0, 6.0, 0.5) / 100
    dividend_yield = st.slider("Dividend Yield (%)", 0.0, 5.0, 1.25, 0.25) / 100
    max_expiries = st.slider("Max Expiries to Fetch", 1, 8, 4, 1)
    
    # Fetch data button
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared! Data will be refreshed.")

# Main content
# Fetch and process data
with st.spinner("🔍 Fetching market data..."):
    df_raw = fetch_option_chain_data(symbol, max_expiries)

if df_raw is not None and not df_raw.empty:
    with st.spinner("⚙️ Processing market data..."):
        df_options, df_pcr, df_crash_probs, df_puts = process_option_data(
            df_raw, risk_free_rate, dividend_yield
        )
    
    if df_options is not None:
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        if not df_options.empty:
            current_spot = df_options['currentSpotPrice'].iloc[0]
            total_options = len(df_options)
            unique_expiries = df_options['expiryDate'].nunique()
            
            with col1:
                #st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
                st.metric("Current Spot", f"₹{current_spot:,.0f}")
                #st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                #st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
                st.metric("Total Options", f"{total_options:,}")
                #st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                #st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
                st.metric("Expiry Dates", unique_expiries)
                #st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                if not df_pcr.empty:
                    avg_pcr_vol = df_pcr['PCR_Volume'].mean()
                    #st.markdown('<div class="metric-highlight">', unsafe_allow_html=True)
                    st.metric("Avg PCR (Vol)", f"{avg_pcr_vol:.2f}" if pd.notna(avg_pcr_vol) else "N/A")
                    #st.markdown('</div>', unsafe_allow_html=True)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "PCR Analysis", 
            "IV Smile", 
            "Crash Probabilities", 
            "Term structure of Implied Volatility",
            "Data Overview"
        ])
        
        with tab1:
            st.markdown('<h2 class="section-header">Put-Call Ratio Analysis</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
            <strong>🎯 Market Surveillance Insight:</strong><br>
            Put-Call Ratio (Volume-based) reveals active trading sentiment and positioning. High PCR (>1.2) suggests bearish sentiment or hedging activity, 
            while low PCR (<0.8) indicates bullish sentiment. PCR changes across expiries show how sentiment evolves over time.
            </div>
            """, unsafe_allow_html=True)
            
            if not df_pcr.empty:
                # Volume-based PCR chart only
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=df_pcr['daysToExpiry'],
                        y=df_pcr['PCR_Volume'],
                        name='PCR (Volume)',
                        marker_color='#5E81AC',
                        text=df_pcr['PCR_Volume'].round(2),
                        textposition='outside'
                    )
                )
                
                # Add reference line
                fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.7)
                
                fig.update_layout(
                    title="<b>Put-Call Ratio (Volume-based) Across Expiries</b>",
                    xaxis_title="Days to Expiry",
                    yaxis_title="PCR (Volume)",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                avg_pcr_vol = df_pcr['PCR_Volume'].mean()
                
                # Market sentiment based on volume PCR
                sentiment = "Bearish" if avg_pcr_vol > 1.2 else "Bullish" if avg_pcr_vol < 0.8 else "Neutral"
                color = "#BF616A" if sentiment == "Bearish" else "#A3BE8C" if sentiment == "Bullish" else "#EBCB8B"
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 1rem; border-radius: 6px; text-align: center;">
                    <strong>Market Sentiment: {sentiment}</strong><br>
                    Average PCR (Volume): {avg_pcr_vol:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    trading_bias = "Put Heavy Trading" if avg_pcr_vol > 1.0 else "Call Heavy Trading"
                    st.markdown(f"""
                    <div style="background-color: #D8DEE9; padding: 1rem; border-radius: 6px; text-align: center;">
                    <strong>Trading Activity: {trading_bias}</strong><br>
                    Current Market Bias
                    </div>
                    """, unsafe_allow_html=True)
                
                # PCR table (volume-based only)
                st.markdown("**📊 Volume-based PCR Details**")
                display_pcr = df_pcr[['expiryDate', 'daysToExpiry', 'PCR_Volume', 'Total_Vol_Puts', 'Total_Vol_Calls']].copy()
                display_pcr['expiryDate'] = display_pcr['expiryDate'].dt.strftime('%d-%b-%Y')
                display_pcr = display_pcr.round(3)
                display_pcr = display_pcr.rename(columns={
                    'expiryDate': 'Expiry Date',
                    'daysToExpiry': 'Days to Expiry',
                    'PCR_Volume': 'PCR (Volume)',
                    'Total_Vol_Puts': 'Put Volume',
                    'Total_Vol_Calls': 'Call Volume'
                })
                st.dataframe(display_pcr, use_container_width=True, hide_index=True)
            else:
                st.warning("No PCR data available for analysis")
        
        with tab2:
            st.markdown('<h2 class="section-header">Implied Volatility Smile</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
            <strong>🎯 Market Surveillance Insight:</strong><br>
            IV Smile reveals market expectations of volatility across strike prices. Steep smiles indicate higher demand for out-of-money options, 
            suggesting hedging activity or directional bets. Asymmetric smiles (skew) often indicate crash fears or bullish positioning.
            </div>
            """, unsafe_allow_html=True)
            
            if not df_options.empty:
                # Get nearest two expiries
                nearest_expiries = sorted(df_options['expiryDate'].unique())[:2]
                
                fig = make_subplots(
                    rows=len(nearest_expiries), 
                    cols=1,
                    subplot_titles=[f"Expiry: {exp.strftime('%d-%b-%Y')} ({(exp - dt.datetime.now()).days} days)" 
                                  for exp in nearest_expiries],
                    vertical_spacing=0.15
                )
                
                for i, expiry in enumerate(nearest_expiries):
                    expiry_data = df_options[df_options['expiryDate'] == expiry]
                    current_spot = expiry_data['currentSpotPrice'].iloc[0]
                    
                    # Filter and sort data
                    ce_data = expiry_data[
                        (expiry_data['optionType'] == 'CE') & 
                        (expiry_data['impliedVolatility'] > 0) & 
                        (expiry_data['impliedVolatility'] < 200)
                    ].sort_values('strikePrice')
                    
                    pe_data = expiry_data[
                        (expiry_data['optionType'] == 'PE') & 
                        (expiry_data['impliedVolatility'] > 0) & 
                        (expiry_data['impliedVolatility'] < 200)
                    ].sort_values('strikePrice')
                    
                    # Plot calls
                    if not ce_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=ce_data['strikePrice'],
                                y=ce_data['impliedVolatility'],
                                mode='markers+lines',
                                name=f'Calls' if i == 0 else None,
                                marker=dict(color='#A3BE8C', size=6),
                                line=dict(color='#A3BE8C', width=2),
                                showlegend=(i == 0)
                            ),
                            row=i+1, col=1
                        )
                    
                    # Plot puts
                    if not pe_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=pe_data['strikePrice'],
                                y=pe_data['impliedVolatility'],
                                mode='markers+lines',
                                name=f'Puts' if i == 0 else None,
                                marker=dict(color='#BF616A', size=6),
                                line=dict(color='#BF616A', width=2),
                                showlegend=(i == 0)
                            ),
                            row=i+1, col=1
                        )
                    
                    # Spot price line
                    fig.add_vline(
                        x=current_spot,
                        line_dash="dash",
                        line_color="#3B4252",
                        annotation_text="Spot",
                        annotation_position="top",
                        row=i+1, col=1
                    )
                    
                    fig.update_xaxes(title_text="Strike Price", row=i+1, col=1)
                    fig.update_yaxes(title_text="Implied Volatility (%)", row=i+1, col=1)
                
                fig.update_layout(
                    height=400 * len(nearest_expiries),
                    title_text="<b>Implied Volatility Smile Analysis</b>",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # IV Skew Analysis
                st.markdown("**📈 Volatility Skew Analysis**")
                
                skew_data = []
                for expiry in nearest_expiries:
                    expiry_data = df_options[df_options['expiryDate'] == expiry]
                    current_spot = expiry_data['currentSpotPrice'].iloc[0]
                    
                    # Calculate ATM, OTM Put, and OTM Call IVs
                    atm_strike = expiry_data.loc[(expiry_data['strikePrice'] - current_spot).abs().idxmin(), 'strikePrice']
                    
                    atm_call_iv = expiry_data[
                        (expiry_data['strikePrice'] == atm_strike) & 
                        (expiry_data['optionType'] == 'CE')
                    ]['impliedVolatility'].iloc[0] if len(expiry_data[
                        (expiry_data['strikePrice'] == atm_strike) & 
                        (expiry_data['optionType'] == 'CE')
                    ]) > 0 else np.nan
                    
                    # OTM Put (below spot)
                    otm_put_data = expiry_data[
                        (expiry_data['strikePrice'] < current_spot * 0.95) & 
                        (expiry_data['optionType'] == 'PE') & 
                        (expiry_data['impliedVolatility'] > 0)
                    ]
                    otm_put_iv = otm_put_data['impliedVolatility'].mean() if not otm_put_data.empty else np.nan
                    
                    # OTM Call (above spot)
                    otm_call_data = expiry_data[
                        (expiry_data['strikePrice'] > current_spot * 1.05) & 
                        (expiry_data['optionType'] == 'CE') & 
                        (expiry_data['impliedVolatility'] > 0)
                    ]
                    otm_call_iv = otm_call_data['impliedVolatility'].mean() if not otm_call_data.empty else np.nan
                    
                    put_skew = otm_put_iv - atm_call_iv if pd.notna(otm_put_iv) and pd.notna(atm_call_iv) else np.nan
                    call_skew = otm_call_iv - atm_call_iv if pd.notna(otm_call_iv) and pd.notna(atm_call_iv) else np.nan
                    
                    skew_data.append({
                        'Expiry': expiry.strftime('%d-%b-%Y'),
                        'Days': (expiry - dt.datetime.now()).days,
                        'ATM_IV': atm_call_iv,
                        'OTM_Put_IV': otm_put_iv,
                        'OTM_Call_IV': otm_call_iv,
                        'Put_Skew': put_skew,
                        'Call_Skew': call_skew
                    })
                
                df_skew = pd.DataFrame(skew_data)
                if not df_skew.empty:
                    df_skew = df_skew.round(2)
                    st.dataframe(df_skew, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    avg_put_skew = df_skew['Put_Skew'].mean()
                    if pd.notna(avg_put_skew):
                        if avg_put_skew > 2:
                            skew_interpretation = "High put skew indicates significant crash protection demand"
                            color = "#BF616A"
                        elif avg_put_skew < -2:
                            skew_interpretation = "Negative put skew suggests complacency or bullish positioning"
                            color = "#EBCB8B"
                        else:
                            skew_interpretation = "Normal volatility skew levels"
                            color = "#A3BE8C"
                        
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 1rem; border-radius: 6px; margin-top: 1rem;">
                        <strong>Skew Interpretation:</strong> {skew_interpretation}<br>
                        Average Put Skew: {avg_put_skew:.2f}%
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No implied volatility data available")
        
        with tab3:
            st.markdown('<h2 class="section-header">Crash Probability Analysis</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
            <strong>🎯 Market Surveillance Insight:</strong><br>
            Crash probabilities derived from put option pricing reveal market-implied expectations of significant downward moves. 
            Higher probabilities for large drops indicate elevated tail risk concerns. Compare near-term vs longer-term probabilities 
            to understand how crash fears evolve over time.
            </div>
            """, unsafe_allow_html=True)
            
            if not df_crash_probs.empty:
                # Format data for plotting
                plot_data = df_crash_probs.copy()
                plot_data['crashProbability'] *= 100
                plot_data['crashThresholdPct'] = (plot_data['crashThresholdPct'].abs() * 100).astype(str) + '% Drop'
                plot_data['expiryDate_str'] = plot_data['expiryDate'].dt.strftime('%d-%b-%Y')
                plot_data['daysToExpiry'] = plot_data['expiryDate'].apply(lambda x: (x - dt.datetime.now()).days)
                
                # Create grouped bar chart
                fig = px.bar(
                    plot_data,
                    x='crashThresholdPct',
                    y='crashProbability',
                    color='expiryDate_str',
                    barmode='group',
                    labels={
                        "crashThresholdPct": "Market Drop Threshold",
                        "crashProbability": "Estimated Probability (%)",
                        "expiryDate_str": "Expiry Date"
                    },
                    title="<b>Market-Implied Crash Probabilities (1st & 4th Expiry)</b>",
                    color_discrete_sequence=['#5E81AC', '#BF616A']
                )
                
                fig.update_layout(
                    xaxis={'categoryorder':'array', 'categoryarray': ['5.0% Drop', '10.0% Drop', '15.0% Drop', '20.0% Drop']},
                    legend_title_text='Expiry Date',
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk Assessment
                st.markdown("**🚨 Risk Assessment**")
                
                # Calculate risk metrics
                near_term_data = plot_data[plot_data['daysToExpiry'] == plot_data['daysToExpiry'].min()]
                longer_term_data = plot_data[plot_data['daysToExpiry'] == plot_data['daysToExpiry'].max()]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if not near_term_data.empty:
                        avg_crash_prob_10 = near_term_data[near_term_data['crashThresholdPct'] == '5.0% Drop']['crashProbability'].iloc[0] if len(near_term_data[near_term_data['crashThresholdPct'] == '5.0% Drop']) > 0 else 0
                        risk_level = "High" if avg_crash_prob_10 > 15 else "Medium" if avg_crash_prob_10 > 8 else "Low"
                        color = "#BF616A" if risk_level == "High" else "#EBCB8B" if risk_level == "Medium" else "#A3BE8C"
                        
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 1rem; border-radius: 6px;">
                        <strong>Near-term Risk Level: {risk_level}</strong><br>
                        5% Drop Probability: {avg_crash_prob_10:.1f}%<br>
                        Days to Expiry: {near_term_data['daysToExpiry'].iloc[0]}
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if not longer_term_data.empty and len(plot_data['daysToExpiry'].unique()) > 1:
                        avg_crash_prob_10_long = longer_term_data[longer_term_data['crashThresholdPct'] == '5.0% Drop']['crashProbability'].iloc[0] if len(longer_term_data[longer_term_data['crashThresholdPct'] == '5.0% Drop']) > 0 else 0
                        
                        st.markdown(f"""
                        <div style="background-color: #D8DEE9; padding: 1rem; border-radius: 6px;">
                        <strong>Longer-term Outlook</strong><br>
                        5% Drop Probability: {avg_crash_prob_10_long:.1f}%<br>
                        Days to Expiry: {longer_term_data['daysToExpiry'].iloc[0]}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Detailed probability table
                st.markdown("**📊 Detailed Crash Probabilities**")
                display_crash = plot_data[[
                    'expiryDate_str', 'daysToExpiry', 'crashThresholdPct', 'crashProbability', 
                    'spotPrice', 'targetStrike'
                ]].rename(columns={
                    'expiryDate_str': 'Expiry',
                    'daysToExpiry': 'Days',
                    'crashThresholdPct': 'Threshold',
                    'crashProbability': 'Probability (%)',
                    'spotPrice': 'Spot Price',
                    'targetStrike': 'Target Strike'
                }).round(2)
                
                st.dataframe(display_crash, use_container_width=True, hide_index=True)
                
                # Warning for high probabilities
                max_prob = plot_data['crashProbability'].max()
                if max_prob > 20:
                    st.markdown("""
                    <div class="warning-box">
                    <strong>⚠️ High Tail Risk Alert:</strong> Market is pricing elevated probability of significant downward moves. 
                    Consider increased hedging or risk management measures.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Insufficient data to calculate crash probabilities. Need OTM put options data.")
        
        with tab4:
            st.markdown('<h2 class="section-header">Term Structure of IV</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
            <strong>🎯 Market Surveillance Insight:</strong><br>
            The IV term structure reveals how volatility expectations change over time. Upward sloping curves suggest near-term stability 
            with longer-term uncertainty. Inverted curves often indicate immediate event risk or market stress.
            </div>
            """, unsafe_allow_html=True)
            
            if not df_options.empty:
                # Calculate term structure data
                term_structure_data = []
                
                for expiry in df_options['expiryDate'].unique():
                    expiry_data = df_options[df_options['expiryDate'] == expiry]
                    
                    # Calculate weighted average IV for calls and puts
                    ce_data = expiry_data[(expiry_data['optionType'] == 'CE') & (expiry_data['impliedVolatility'] > 0)]
                    pe_data = expiry_data[(expiry_data['optionType'] == 'PE') & (expiry_data['impliedVolatility'] > 0)]
                    
                    if not ce_data.empty and not pe_data.empty:
                        # Weight by open interest
                        ce_weighted_iv = np.average(
                            ce_data['impliedVolatility'], 
                            weights=ce_data['openInterest'].fillna(1)
                        )
                        
                        pe_weighted_iv = np.average(
                            pe_data['impliedVolatility'], 
                            weights=pe_data['openInterest'].fillna(1)
                        )
                        
                        # Average of call and put weighted IV
                        avg_weighted_iv = (ce_weighted_iv + pe_weighted_iv) / 2
                        
                        days_to_expiry = (expiry - dt.datetime.now()).days
                        
                        term_structure_data.append({
                            'expiryDate': expiry,
                            'daysToExpiry': days_to_expiry,
                            'avgWeightedIV': avg_weighted_iv,
                            'callWeightedIV': ce_weighted_iv,
                            'putWeightedIV': pe_weighted_iv,
                            'ivSkew': pe_weighted_iv - ce_weighted_iv
                        })
                
                if term_structure_data:
                    df_term = pd.DataFrame(term_structure_data).sort_values('daysToExpiry')
                    
                    # Term structure chart
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df_term['daysToExpiry'],
                            y=df_term['avgWeightedIV'],
                            mode='markers+lines',
                            name='Average IV Term Structure',
                            line=dict(color='#5E81AC', width=3),
                            marker=dict(size=8, color='#5E81AC')
                        )
                    )
                    
                    fig.update_layout(
                        title="<b>Implied Volatility Term Structure</b>",
                        xaxis_title="Days to Expiry",
                        yaxis_title="Implied Volatility (%)",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Term structure interpretation
                    if len(df_term) >= 2:
                        near_iv = df_term.iloc[0]['avgWeightedIV']
                        far_iv = df_term.iloc[-1]['avgWeightedIV']
                        slope = (far_iv - near_iv) / (df_term.iloc[-1]['daysToExpiry'] - df_term.iloc[0]['daysToExpiry']) * 30  # Per 30 days
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if slope > 0.5:
                                structure_type = "Steep Upward Slope"
                                interpretation = "Market expects volatility to increase over time"
                                color = "#EBCB8B"
                            elif slope < -0.5:
                                structure_type = "Inverted (Downward Slope)"
                                interpretation = "Near-term volatility concerns or event risk"
                                color = "#BF616A"
                            else:
                                structure_type = "Flat Structure"
                                interpretation = "Stable volatility expectations across time"
                                color = "#A3BE8C"
                            
                            st.markdown(f"""
                            <div style="background-color: {color}; padding: 1rem; border-radius: 6px;">
                            <strong>Term Structure: {structure_type}</strong><br>
                            Slope: {slope:.2f}% per 30 days<br>
                            {interpretation}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            avg_skew = df_term['ivSkew'].mean()
                            skew_interpretation = "High put demand" if avg_skew > 2 else "Balanced" if abs(avg_skew) <= 2 else "Call heavy"
                            
                            st.markdown(f"""
                            <div style="background-color: #D8DEE9; padding: 1rem; border-radius: 6px;">
                            <strong>Average IV Skew: {avg_skew:.2f}%</strong><br>
                            Market Bias: {skew_interpretation}<br>
                            Put-Call IV Difference
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Term structure data table
                    st.markdown("**📊 Term Structure Details**")
                    display_term = df_term.copy()
                    display_term['expiryDate'] = display_term['expiryDate'].dt.strftime('%d-%b-%Y')
                    display_term = display_term[['expiryDate', 'daysToExpiry', 'avgWeightedIV', 'ivSkew']].round(2)
                    display_term = display_term.rename(columns={
                        'expiryDate': 'Expiry Date',
                        'daysToExpiry': 'Days to Expiry',
                        'avgWeightedIV': 'Average IV (%)',
                        'ivSkew': 'IV Skew (%)'
                    })
                    st.dataframe(display_term, use_container_width=True, hide_index=True)
                else:
                    st.warning("Insufficient data for term structure analysis")
            else:
                st.warning("No options data available")
        
        with tab5:
            st.markdown('<h2 class="section-header">Data Overview</h2>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="interpretation-box">
            <strong>📋 Raw Data Access:</strong><br>
            Detailed options data for further analysis. Select specific expiry dates to examine individual option contracts, 
            pricing, and trading activity.
            </div>
            """, unsafe_allow_html=True)
            
            # Expiry selection for detailed view
            selected_expiry = st.selectbox(
                "Select Expiry for Detailed View",
                df_options['expiryDate'].dt.strftime('%d-%b-%Y').unique(),
                key="data_overview_expiry"
            )
            
            expiry_data = df_options[
                df_options['expiryDate'].dt.strftime('%d-%b-%Y') == selected_expiry
            ].copy()
            
            # Summary statistics for selected expiry
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_oi = expiry_data['openInterest'].sum()
                st.metric("Total OI", f"{total_oi:,.0f}")
            
            with col2:
                total_volume = expiry_data['totalTradedVolume'].sum()
                st.metric("Total Volume", f"{total_volume:,.0f}")
            
            with col3:
                avg_iv = expiry_data['impliedVolatility'].mean()
                st.metric("Avg IV", f"{avg_iv:.1f}%" if pd.notna(avg_iv) else "N/A")
            
            with col4:
                days_to_exp = (pd.to_datetime(selected_expiry, format='%d-%b-%Y') - dt.datetime.now()).days
                st.metric("Days to Expiry", days_to_exp)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📞 Call Options (CE)**")
                ce_data = expiry_data[expiry_data['optionType'] == 'CE'].sort_values('strikePrice')
                if not ce_data.empty:
                    display_cols = ['strikePrice', 'lastPrice', 'impliedVolatility', 
                                  'openInterest', 'totalTradedVolume', 'change']
                    ce_display = ce_data[display_cols].round(2)
                    st.dataframe(ce_display, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**📉 Put Options (PE)**")
                pe_data = expiry_data[expiry_data['optionType'] == 'PE'].sort_values('strikePrice')
                if not pe_data.empty:
                    display_cols = ['strikePrice', 'lastPrice', 'impliedVolatility', 
                                  'openInterest', 'totalTradedVolume', 'change']
                    pe_display = pe_data[display_cols].round(2)
                    st.dataframe(pe_display, use_container_width=True, hide_index=True)
            
            # Export functionality
            st.markdown("---")
            st.markdown("**📊 Data Export**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export Selected Expiry", use_container_width=True):
                    csv_buffer = io.StringIO()
                    expiry_data.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{symbol}_{selected_expiry}_options.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📥 Export All Data", use_container_width=True):
                    csv_buffer = io.StringIO()
                    df_options.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download All CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"{symbol}_all_options_{dt.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    else:
        st.error("❌ Failed to process option data. Please check the data source and try again.")
else:
    st.error("❌ Failed to fetch option chain data. Please check your internet connection and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>NSE Options Market Surveillance Dashboard</strong> | Real-time data from NSE India</p>
    <p><small>⚠️ Disclaimer: This dashboard is for educational and analytical purposes only. Please consult with a qualified financial advisor before making investment decisions.</small></p>
</div>
""", unsafe_allow_html=True)
