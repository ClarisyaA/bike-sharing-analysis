import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import Patch
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #f7f9fc; }
    .main-title {
        font-size: 1.8rem; font-weight: 700; color: #1a3a6b;
        text-align: center; margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 0.9rem; color: #666; text-align: center;
        margin-top: 0; margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #ffffff; border-radius: 8px; padding: 14px 18px;
        text-align: center; border-top: 3px solid #1a3a6b;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label { font-size: 0.72rem; color: #777; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1a3a6b; margin-top: 4px; }
    .section-header {
        font-size: 0.95rem; font-weight: 700; color: #1a3a6b;
        border-bottom: 2px solid #d0daea; padding-bottom: 5px;
        margin-top: 1.2rem; margin-bottom: 0.8rem;
        text-transform: uppercase; letter-spacing: 0.4px;
    }
    .insight-box {
        background: #f0f5ff; border-left: 4px solid #1a3a6b;
        border-radius: 4px; padding: 12px 16px; margin-top: 12px;
        font-size: 0.88rem; color: #333; line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 0.85rem; color: #555;
        border-radius: 6px 6px 0 0; padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] { color: #1a3a6b !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MATPLOTLIB STYLE
# ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.dpi': 110,
})

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    hour_path = os.path.join(base, "main_data.csv")
    day_path  = os.path.join(base, "..", "data", "day.csv")

    hour_df = pd.read_csv(hour_path, parse_dates=['dteday'])
    day_df  = pd.read_csv(day_path,  parse_dates=['dteday'])

    season_map  = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weather_map = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    weekday_map = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
    month_map   = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                   7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    year_map    = {0: 2011, 1: 2012}

    for df in [day_df, hour_df]:
        df['season_label']  = df['season'].map(season_map)
        df['weather_label'] = df['weathersit'].map(weather_map)
        df['weekday_label'] = df['weekday'].map(weekday_map)
        df['year_actual']   = df['yr'].map(year_map)

    day_df['month_label']        = day_df['mnth'].map(month_map)
    day_df['temp_celsius']       = day_df['temp'] * 41
    day_df['hum_percent']        = day_df['hum'] * 100
    day_df['windspeed_kmh']      = day_df['windspeed'] * 67
    day_df['workingday_label']   = day_df['workingday'].map({0: 'Holiday/Weekend', 1: 'Working Day'})
    day_df['usage_category']     = pd.cut(
        day_df['cnt'],
        bins=[0, 2000, 4000, 6000, 9000],
        labels=['Low (0-2K)', 'Medium (2K-4K)', 'High (4K-6K)', 'Very High (6K+)'],
        include_lowest=True
    )

    hour_df['workingday_label'] = hour_df['workingday'].map({0: 'Holiday/Weekend', 1: 'Working Day'})
    hour_df['temp_celsius']     = hour_df['temp'] * 41

    return day_df, hour_df

day_df, hour_df = load_data()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
st.sidebar.markdown("## Bike Sharing Dashboard")
st.sidebar.markdown("Capital Bikeshare, Washington D.C.")
st.sidebar.markdown("---")

year_filter = st.sidebar.multiselect(
    "Year", options=[2011, 2012], default=[2011, 2012]
)
season_filter = st.sidebar.multiselect(
    "Season", options=['Spring', 'Summer', 'Fall', 'Winter'],
    default=['Spring', 'Summer', 'Fall', 'Winter']
)
weather_filter = st.sidebar.multiselect(
    "Weather", options=['Clear', 'Mist/Cloudy', 'Light Rain/Snow'],
    default=['Clear', 'Mist/Cloudy', 'Light Rain/Snow']
)

day_filtered = day_df[
    day_df['year_actual'].isin(year_filter) &
    day_df['season_label'].isin(season_filter) &
    day_df['weather_label'].isin(weather_filter)
].copy()

hour_filtered = hour_df[
    hour_df['year_actual'].isin(year_filter) &
    hour_df['season_label'].isin(season_filter) &
    hour_df['weather_label'].isin(weather_filter)
].copy()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected data:** {len(day_filtered):,} days")
st.sidebar.markdown("**Source:** Capital Bikeshare, Washington D.C. 2011–2012")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">Bike Sharing Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Capital Bikeshare, Washington D.C. — Analysis of Bike Rental Patterns 2011–2012</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# KPI METRICS
# ──────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_rentals = day_filtered['cnt'].sum()
avg_daily     = day_filtered['cnt'].mean()
total_casual  = day_filtered['casual'].sum()
total_reg     = day_filtered['registered'].sum()
peak_val      = day_filtered['cnt'].max() if len(day_filtered) > 0 else 0

metrics = [
    ("Total Rentals", f"{total_rentals:,.0f}"),
    ("Avg per Day", f"{avg_daily:,.0f}"),
    ("Casual Users", f"{total_casual:,.0f}"),
    ("Registered Users", f"{total_reg:,.0f}"),
    ("Peak Day", f"{peak_val:,.0f}"),
]
for col, (label, value) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
BLUE     = '#1a3a6b'
ORANGE   = '#d45f00'
LBLUE    = '#4a7cc9'
LGRAY    = '#8fa8c8'
CAT_COL  = ['#c0392b', '#e67e22', '#27ae60', '#2980b9']
SEA_COL  = {'Spring': '#a8d5a2', 'Summer': '#f7e87b', 'Fall': '#f0a67a', 'Winter': '#90bfe0'}

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Question 1: Weather & Temperature",
    "Question 2: Peak Hours",
    "Advanced Analysis"
])

# ───────── TAB 1: OVERVIEW ─────────
with tab1:
    st.markdown('<div class="section-header">Daily Rental Trend</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(day_filtered['dteday'], day_filtered['cnt'],
                color=LBLUE, linewidth=0.9, alpha=0.6)
        ax.fill_between(day_filtered['dteday'], day_filtered['cnt'], alpha=0.1, color=BLUE)
        if len(day_filtered) >= 30:
            rolling = day_filtered.set_index('dteday')['cnt'].rolling(30).mean()
            ax.plot(rolling.index, rolling.values, color=ORANGE, linewidth=2.2,
                    label='30-day moving avg', zorder=5)
        ax.set_title('Daily Bike Rentals (2011–2012)', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Rentals', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<div class="section-header">Season Share</div>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(3.8, 3.5))
        season_data = day_filtered.groupby('season_label')['cnt'].sum()
        s_order = [s for s in ['Spring', 'Summer', 'Fall', 'Winter'] if s in season_data.index]
        colors_pie = [SEA_COL[s] for s in s_order]
        ax2.pie(
            season_data.reindex(s_order), labels=s_order, colors=colors_pie,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 8},
            wedgeprops={'linewidth': 1.2, 'edgecolor': 'white'}
        )
        ax2.set_title('Rental Share by Season', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown('<div class="section-header">Year-over-Year Comparison</div>', unsafe_allow_html=True)
    col_y1, col_y2 = st.columns(2)
    for yr, col in zip([2011, 2012], [col_y1, col_y2]):
        if yr in year_filter:
            yr_data = day_filtered[day_filtered['year_actual'] == yr]
            with col:
                st.metric(
                    label=f"Total {yr}",
                    value=f"{yr_data['cnt'].sum():,.0f}",
                    delta=f"Avg {yr_data['cnt'].mean():,.0f}/day"
                )

# ───────── TAB 2: CUACA & SUHU ─────────
with tab2:
    st.markdown('<div class="section-header">Question 1: How do weather conditions and temperature affect daily bike rentals?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        weather_order  = ['Clear', 'Mist/Cloudy', 'Light Rain/Snow']
        weather_colors = [BLUE, LBLUE, LGRAY]
        w_data = day_filtered.groupby('weather_label')['cnt'].mean()
        w_data = w_data.reindex([w for w in weather_order if w in w_data.index])
        bars = ax.bar(w_data.index, w_data.values, color=weather_colors[:len(w_data)],
                      edgecolor='white', width=0.5)
        for bar, val in zip(bars, w_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title('Average Rentals by Weather Condition', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg rentals/day', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_ylim(0, (max(w_data.values) * 1.2) if len(w_data) else 6000)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sc = ax.scatter(
            day_filtered['temp'] * 41, day_filtered['cnt'],
            c=day_filtered['weathersit'], cmap='Blues',
            alpha=0.5, s=22, edgecolors='none'
        )
        if len(day_filtered) > 2:
            temps = day_filtered['temp'] * 41
            z = np.polyfit(temps, day_filtered['cnt'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(temps.min(), temps.max(), 100)
            r = temps.corr(day_filtered['cnt'])
            ax.plot(x_line, p(x_line), color=ORANGE, linewidth=2, linestyle='--',
                    label=f'Trend (r={r:.2f})')
        plt.colorbar(sc, ax=ax, label='Weather (1=Clear, 3=Rain)')
        ax.set_title('Temperature (°C) vs Rentals', fontsize=11, fontweight='bold')
        ax.set_xlabel('Temperature (°C)', fontsize=9)
        ax.set_ylabel('Total Rentals (cnt)', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(axis='both', linestyle='--', alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        s_order2 = [s for s in ['Spring', 'Summer', 'Fall', 'Winter']
                    if s in day_filtered['season_label'].values]
        w_order2 = [w for w in ['Clear', 'Mist/Cloudy', 'Light Rain/Snow']
                    if w in day_filtered['weather_label'].values]
        if s_order2 and w_order2:
            hm_data = (day_filtered
                       .groupby(['season_label', 'weather_label'])['cnt']
                       .mean().unstack(fill_value=0))
            hm_data = hm_data.reindex(index=s_order2, columns=w_order2, fill_value=0)
            sns.heatmap(hm_data, annot=True, fmt='.0f', cmap='Blues', ax=ax,
                        linewidths=0.5, linecolor='white',
                        annot_kws={'size': 10, 'weight': 'bold'})
        ax.set_title('Avg Rentals: Season x Weather', fontsize=11, fontweight='bold')
        ax.set_xlabel('Weather Condition', fontsize=9)
        ax.set_ylabel('Season', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        w_groups = [day_filtered[day_filtered['weather_label'] == w]['cnt'].values
                    for w in weather_order if w in day_filtered['weather_label'].values]
        w_labels = [w for w in weather_order if w in day_filtered['weather_label'].values]
        if w_groups:
            bplots = ax.boxplot(w_groups, labels=w_labels, patch_artist=True,
                                medianprops=dict(color='black', linewidth=2))
            for patch, color in zip(bplots['boxes'], weather_colors[:len(w_groups)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title('Rental Distribution by Weather', fontsize=11, fontweight='bold')
        ax.set_ylabel('Rentals (cnt)', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.set_xticklabels(w_labels, rotation=10, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""<div class="insight-box">
    <b>Insight:</b> Clear weather yields an average of <b>~4,877 rentals/day</b>,
    while light rain/snow results in only <b>~1,803/day</b> — a drop of <b>63%</b>.
    Temperature has a strong positive correlation (r = 0.63) with rental volume.
    The optimal combination is Fall + Clear weather at an average temperature of ~22°C.
    Operators are advised to integrate weather forecasts into fleet management systems.
    </div>""", unsafe_allow_html=True)

# ───────── TAB 3: POLA JAM ─────────
with tab3:
    st.markdown('<div class="section-header">Question 2: What are the peak rental hours and how do they differ between working days and holidays?</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        colors_wk = {'Working Day': BLUE, 'Holiday/Weekend': ORANGE}
        for label, grp in hour_filtered.groupby('workingday_label'):
            avg = grp.groupby('hr')['cnt'].mean()
            ax.plot(avg.index, avg.values, marker='o', markersize=3,
                    label=label, color=colors_wk.get(label, 'gray'), linewidth=2)
        ax.set_title('Hourly Rental Pattern\n(Working Day vs Holiday)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Hour (0–23)', fontsize=9)
        ax.set_ylabel('Avg Rentals/Hour', fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)], fontsize=7.5, rotation=30)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.axvspan(7, 9, alpha=0.07, color=BLUE)
        ax.axvspan(16, 19, alpha=0.07, color=BLUE)
        ax.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        wd_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hm_hw = hour_filtered.groupby(['weekday_label', 'hr'])['cnt'].mean().unstack(fill_value=0)
        hm_hw = hm_hw.reindex([w for w in wd_order if w in hm_hw.index])
        if not hm_hw.empty:
            sns.heatmap(hm_hw, cmap='Blues', ax=ax, linewidths=0,
                        xticklabels=2, cbar_kws={'label': 'Avg rentals/hour'})
        ax.set_title('Rental Heatmap: Hour x Weekday', fontsize=11, fontweight='bold')
        ax.set_xlabel('Hour', fontsize=9)
        ax.set_ylabel('Day', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        fig, axes_sub = plt.subplots(1, 2, figsize=(7, 3.5), sharey=False)
        titles_sub = ['Working Day', 'Holiday / Weekend']
        wd_vals = [1, 0]
        for ax_s, wd_val, title_s in zip(axes_sub, wd_vals, titles_sub):
            sub = (hour_filtered[hour_filtered['workingday'] == wd_val]
                   .groupby('hr')[['casual', 'registered']].mean())
            ax_s.fill_between(sub.index, sub['casual'], alpha=0.35, color='#c0392b', label='Casual')
            ax_s.fill_between(sub.index, sub['registered'], alpha=0.35, color=BLUE, label='Registered')
            ax_s.plot(sub.index, sub['casual'], color='#c0392b', linewidth=1.5)
            ax_s.plot(sub.index, sub['registered'], color=BLUE, linewidth=1.5)
            ax_s.set_title(title_s, fontsize=9, fontweight='bold')
            ax_s.set_xlabel('Hour', fontsize=8)
            ax_s.set_xticks(range(0, 24, 4))
            ax_s.legend(fontsize=7, framealpha=0.7)
            ax_s.grid(axis='y', linestyle='--', alpha=0.35)
            ax_s.spines[['top', 'right']].set_visible(False)
        axes_sub[0].set_ylabel('Avg rentals/hour', fontsize=8)
        fig.suptitle('Casual vs Registered Users by Hour', fontsize=10, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        wd_order2 = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        wk_colors2 = [BLUE] * 5 + [ORANGE] * 2
        wk_avg = day_filtered.groupby('weekday_label')['cnt'].mean()
        wk_avg = wk_avg.reindex([w for w in wd_order2 if w in wk_avg.index])
        if not wk_avg.empty:
            bars = ax.bar(wk_avg.index, wk_avg.values,
                          color=wk_colors2[:len(wk_avg)], edgecolor='white', width=0.6)
            for bar, val in zip(bars, wk_avg.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                        f'{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        legend_els = [Patch(facecolor=BLUE, label='Working Day'),
                      Patch(facecolor=ORANGE, label='Weekend')]
        ax.legend(handles=legend_els, fontsize=8, framealpha=0.7)
        ax.set_title('Avg Rentals by Day of Week', fontsize=11, fontweight='bold')
        ax.set_ylabel('Avg rentals/day', fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("""<div class="insight-box">
    <b>Insight:</b> Working days show a <b>bimodal pattern</b> with peaks at 08:00 and 17:00 (commuter behavior),
    dominated by <b>registered users</b>. Holidays exhibit a <b>unimodal pattern</b> peaking around 11:00–14:00
    with a higher proportion of <b>casual users</b>. These distinct segments — daily commuters vs recreational
    cyclists — require different fleet distribution strategies.
    </div>""", unsafe_allow_html=True)

# ───────── TAB 4: ANALISIS LANJUTAN ─────────
with tab4:
    st.markdown('<div class="section-header">Advanced Analysis: Manual Clustering by Usage Category</div>', unsafe_allow_html=True)
    st.markdown(
        "Days are grouped using **binning** into 4 usage categories to understand operational "
        "profiles and support data-driven fleet planning decisions."
    )

    cat_order = ['Low (0-2K)', 'Medium (2K-4K)', 'High (4K-6K)', 'Very High (6K+)']

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = (day_filtered['usage_category']
                  .value_counts()
                  .reindex(cat_order, fill_value=0))
        total_days = counts.sum()
        bars = ax.barh(cat_order, counts.values, color=CAT_COL, edgecolor='white', height=0.55)
        for bar, val in zip(bars, counts.values):
            pct = val / total_days * 100 if total_days > 0 else 0
            ax.text(bar.get_width() + 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val} days ({pct:.0f}%)',
                    va='center', fontsize=9, fontweight='bold')
        ax.set_title('Distribution of Days by Usage Category', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Days', fontsize=9)
        ax.set_xlim(0, counts.max() * 1.4 if counts.max() > 0 else 100)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        temp_by_cat = (day_filtered
                       .groupby('usage_category', observed=True)['temp_celsius']
                       .mean()
                       .reindex(cat_order))
        bars2 = ax.bar(cat_order, temp_by_cat.values, color=CAT_COL, edgecolor='white', width=0.5)
        for bar, val in zip(bars2, temp_by_cat.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{val:.1f} C', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title('Avg Temperature by Usage Category', fontsize=11, fontweight='bold')
        ax.set_ylabel('Temperature (C)', fontsize=9)
        ax.set_xticklabels(cat_order, rotation=10, ha='right', fontsize=8)
        ax.set_ylim(0, 30)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-header">Cluster Detail Profile</div>', unsafe_allow_html=True)
    profile = (day_filtered
               .groupby('usage_category', observed=True)
               .agg(
                   Days=('cnt', 'count'),
                   Avg_Rentals=('cnt', 'mean'),
                   Avg_Temp_C=('temp_celsius', 'mean'),
                   Avg_Humidity=('hum_percent', 'mean'),
                   Pct_Clear_Weather=('weathersit', lambda x: (x == 1).mean() * 100),
                   Pct_Working_Day=('workingday', 'mean')
               )
               .reindex(cat_order)
               .round(1)
               .reset_index())
    profile.columns = [
        'Category', 'Days', 'Avg Rentals', 'Avg Temp (C)',
        'Avg Humidity (%)', '% Clear Weather', '% Working Day'
    ]
    st.dataframe(profile, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    sc_data = (day_filtered
               .groupby(['usage_category', 'season_label'], observed=True)
               .size().unstack(fill_value=0))
    sc_data = sc_data.reindex(cat_order)
    s_order3 = ['Spring', 'Summer', 'Fall', 'Winter']
    bottom = np.zeros(len(sc_data))
    for s in s_order3:
        if s in sc_data.columns:
            vals = sc_data[s].fillna(0).values
            ax.bar(sc_data.index, vals, bottom=bottom, label=s,
                   color=SEA_COL[s], edgecolor='white', linewidth=0.6)
            bottom += vals
    ax.set_title('Season Composition by Usage Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Days', fontsize=9)
    ax.set_xticklabels(cat_order, rotation=8, ha='right', fontsize=9)
    ax.legend(title='Season', fontsize=8, title_fontsize=8, framealpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""<div class="insight-box">
    <b>Clustering Insight:</b> The <b>Very High (6K+)</b> category is concentrated in Fall and Summer
    with average temperatures around 25.6°C and mostly clear weather.
    The <b>Low (0-2K)</b> category occurs almost exclusively in Spring at low temperatures (~11°C).
    This clustering can serve as an operational framework: increase available bikes when forecasts
    show high temperatures and clear skies, especially in Fall and Summer.
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Bike Sharing Data Analysis Project — Dicoding | "
    "Data: Capital Bikeshare Washington D.C. 2011–2012</small></center>",
    unsafe_allow_html=True
)
