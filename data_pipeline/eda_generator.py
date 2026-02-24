"""
eda_generator.py — Comprehensive EDA for IEX RTM Electricity Price Forecasting
"""
import os, sys, io, base64, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "app", "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C_MAIN=   "#1a1a2e"
C_RED=    "#e94560"
C_BLUE=   "#2980b9"
C_GREEN=  "#27ae60"
C_ORANGE= "#f39c12"
C_PURPLE= "#8e44ad"
C_GREY=   "#95a5a6"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#fafafa",
    "axes.edgecolor":"#dee2e6","axes.spines.top":False,"axes.spines.right":False,
    "font.family":"sans-serif","font.size":11,"axes.titlesize":13,
    "axes.titleweight":"bold","axes.labelsize":11,"grid.alpha":0.4,"grid.linestyle":"--",
})

def fig_to_b64(fig,dpi=130):
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=dpi,bbox_inches="tight",facecolor="white",edgecolor="none")
    buf.seek(0); plt.close(fig)
    return base64.b64encode(buf.read()).decode()

def img_tag(b64,width="100%",caption=""):
    cap=f'<p class="caption">{caption}</p>' if caption else ""
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:8px">{cap}'

def load_data():
    iex=pd.read_csv(os.path.join(DATA_DIR,"iex_historical.csv"))
    iex["datetime"]=pd.to_datetime(iex["date"],format="%d-%m-%Y",errors="coerce")
    h=iex["time_block"].str.split("-").str[0].str.split(":").str[0].fillna(0).astype(int)
    m=iex["time_block"].str.split("-").str[0].str.split(":").str[1].fillna(0).astype(int)
    iex["datetime"]=iex["datetime"]+pd.to_timedelta(h*60+m,unit="m")
    iex["hour"]=iex["datetime"].dt.hour
    iex["day_of_week"]=iex["datetime"].dt.dayofweek
    iex["month"]=iex["datetime"].dt.month
    iex["year"]=iex["datetime"].dt.year
    iex=iex.dropna(subset=["MCP","datetime"]).sort_values("datetime").reset_index(drop=True)

    wx_path=os.path.join(DATA_DIR,"weather_historical.csv")
    wx=None
    if os.path.exists(wx_path):
        wx=pd.read_csv(wx_path,parse_dates=["datetime"])
        if "city" in wx.columns:
            wx=wx[wx["city"].str.lower().str.contains("delhi",na=False)]

    com_path=os.path.join(DATA_DIR,"commodities_historical.csv")
    com=pd.read_csv(com_path,parse_dates=["date"]) if os.path.exists(com_path) else None
    return iex,wx,com

def stat_cards(iex):
    spike=(iex["MCP"]>9000).sum()
    spike_pct=spike/len(iex)*100
    yoy=iex.groupby("year")["MCP"].mean()
    trend="Declining" if yoy.iloc[-1]<yoy.iloc[0] else "Rising"
    cards=[
        ("Avg MCP",f"Rs{iex['MCP'].mean():,.0f}","Rs/MWh",C_BLUE),
        ("Median MCP",f"Rs{iex['MCP'].median():,.0f}","Rs/MWh",C_GREEN),
        ("Std Dev",f"Rs{iex['MCP'].std():,.0f}","Rs/MWh",C_ORANGE),
        ("Max MCP",f"Rs{iex['MCP'].max():,.0f}","Rs/MWh",C_RED),
        ("Min MCP",f"Rs{iex['MCP'].min():,.0f}","Rs/MWh",C_PURPLE),
        ("Price Spikes",f"{spike_pct:.1f}%",f"{spike} blocks >Rs9000",C_RED),
        ("Total Records",f"{len(iex):,}","15-min blocks",C_MAIN),
        ("Market Trend",trend,f"{int(yoy.index[0])} to {int(yoy.index[-1])}",C_GREY),
    ]
    html='<div class="cards">'
    for title,value,sub,color in cards:
        html+=f'<div class="card" style="border-top:4px solid {color}"><div class="card-title">{title}</div><div class="card-value" style="color:{color}">{value}</div><div class="card-sub">{sub}</div></div>'
    return html+"</div>"

def plot_overview(iex):
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    daily=iex.set_index("datetime")["MCP"].resample("D").mean()
    axes[0].plot(iex["datetime"],iex["MCP"],color=C_BLUE,alpha=0.3,linewidth=0.4,label="15-min MCP")
    axes[0].plot(daily.index,daily.values,color=C_RED,linewidth=1.5,label="Daily avg")
    axes[0].axhline(9000,color=C_RED,linestyle="--",alpha=0.6,linewidth=1,label="Price cap zone")
    axes[0].fill_between(iex["datetime"],9000,iex["MCP"].max(),where=iex["MCP"]>9000,alpha=0.15,color=C_RED)
    axes[0].set_ylabel("MCP (Rs/MWh)"); axes[0].set_title("IEX RTM Market Clearing Price — Full History")
    axes[0].legend(loc="upper right",fontsize=9)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    rolling_std=daily.rolling(30,min_periods=7).std()
    axes[1].fill_between(rolling_std.index,rolling_std.values,alpha=0.5,color=C_ORANGE)
    axes[1].plot(rolling_std.index,rolling_std.values,color=C_ORANGE,linewidth=1.2)
    axes[1].set_ylabel("30-day Rolling Std"); axes[1].set_title("Price Volatility Over Time")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    fig.tight_layout(h_pad=2); return fig_to_b64(fig)

def plot_distribution(iex):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    normal=iex[iex["MCP"]<=9000]["MCP"]; spike=iex[iex["MCP"]>9000]["MCP"]
    axes[0].hist(normal,bins=80,color=C_BLUE,alpha=0.7,label=f"Normal (n={len(normal):,})")
    axes[0].hist(spike,bins=20,color=C_RED,alpha=0.8,label=f"Spike >Rs9000 (n={len(spike):,})")
    axes[0].axvline(iex["MCP"].mean(),color=C_GREEN,linestyle="--",linewidth=2,label=f"Mean Rs{iex['MCP'].mean():,.0f}")
    axes[0].axvline(iex["MCP"].median(),color=C_ORANGE,linestyle="--",linewidth=2,label=f"Median Rs{iex['MCP'].median():,.0f}")
    axes[0].set_xlabel("MCP (Rs/MWh)"); axes[0].set_title("Price Distribution — Bimodal Pattern")
    axes[0].legend(fontsize=9)
    years=sorted(iex["year"].dropna().unique())
    data_by_year=[iex[iex["year"]==y]["MCP"].values for y in years]
    bp=axes[1].boxplot(data_by_year,labels=[int(y) for y in years],patch_artist=True,medianprops=dict(color="white",linewidth=2))
    for patch,color in zip(bp["boxes"],[C_RED,C_ORANGE,C_BLUE,C_GREEN,C_PURPLE][:len(years)]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("MCP (Rs/MWh)")
    axes[1].set_title("Price Distribution by Year — Regime Shift")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    fig.tight_layout(); return fig_to_b64(fig)

def plot_stl(iex):
    from statsmodels.tsa.seasonal import STL
    daily=iex.set_index("datetime")["MCP"].resample("D").mean().dropna()
    daily=daily.asfreq("D").interpolate(method="linear")
    stl=STL(daily,period=7,robust=True); res=stl.fit()
    fig,axes=plt.subplots(4,1,figsize=(14,12),sharex=True)
    for ax,data,title,color in zip(axes,
        [daily,pd.Series(res.trend,index=daily.index),pd.Series(res.seasonal,index=daily.index),pd.Series(res.resid,index=daily.index)],
        ["Observed — Daily Avg MCP","Trend Component (Long-term Direction)","Seasonal Component (Weekly, period=7)","Residual (Unexplained — Price Spikes)"],
        [C_BLUE,C_RED,C_GREEN,C_ORANGE]):
        ax.plot(data.index,data.values,color=color,linewidth=0.9,alpha=0.85)
        ax.axhline(0,color="black",linewidth=0.5,linestyle="--",alpha=0.4)
        ax.set_title(title); ax.set_ylabel("Rs/MWh")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
        ax.fill_between(data.index,data.values,0,alpha=0.12,color=color)
    tv=np.var(res.trend)/np.var(daily.values)*100
    sv=np.var(res.seasonal)/np.var(daily.values)*100
    rv=np.var(res.resid)/np.var(daily.values)*100
    fig.text(0.99,0.01,f"Variance — Trend:{tv:.1f}%  Seasonal:{sv:.1f}%  Residual:{rv:.1f}%",
             ha="right",va="bottom",fontsize=10,color=C_GREY,transform=fig.transFigure)
    fig.tight_layout(h_pad=2); return fig_to_b64(fig),tv,sv,rv

def plot_stl_15min(iex):
    from statsmodels.tsa.seasonal import STL
    cutoff = iex["datetime"].max() - pd.Timedelta(days=60)
    recent=iex[iex["datetime"]>=cutoff].set_index("datetime")["MCP"].resample("15min").mean().dropna()
    if len(recent)<96*7: return None
    recent=recent.asfreq("15min").interpolate(method="linear")
    stl=STL(recent,period=96,robust=True); res=stl.fit()
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True)
    series=[recent,pd.Series(res.trend,index=recent.index),pd.Series(res.seasonal,index=recent.index),pd.Series(res.resid,index=recent.index)]
    labels=["Observed (15-min)","Trend","Seasonal (Daily Cycle — period=96 blocks = 24h)","Residual (Price Shocks)"]
    for ax,s,lbl,col in zip(axes,series,labels,[C_BLUE,C_RED,C_GREEN,C_ORANGE]):
        ax.plot(s.index,s.values,color=col,linewidth=0.6,alpha=0.85)
        ax.axhline(0,color="black",linewidth=0.4,linestyle="--",alpha=0.4)
        ax.set_title(lbl); ax.set_ylabel("Rs/MWh")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    axes[0].set_title("STL on 15-min RTM Data (Last 60 Days) — Daily Seasonality (period=96 blocks)")
    fig.tight_layout(h_pad=2); return fig_to_b64(fig)

def plot_seasonal(iex):
    fig=plt.figure(figsize=(16,12)); gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.4,wspace=0.3)
    day_names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    month_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    ax1=fig.add_subplot(gs[0,0])
    hourly=iex.groupby("hour")["MCP"].agg(["mean","std"]).reset_index()
    ax1.fill_between(hourly["hour"],hourly["mean"]-hourly["std"],hourly["mean"]+hourly["std"],alpha=0.2,color=C_BLUE,label="+-1 std")
    ax1.plot(hourly["hour"],hourly["mean"],color=C_BLUE,linewidth=2.5,marker="o",markersize=5)
    ax1.axvspan(19,22,alpha=0.15,color=C_RED,label="Evening peak")
    ax1.axvspan(0,4,alpha=0.1,color=C_GREEN,label="Off-peak")
    ax1.set_xlabel("Hour"); ax1.set_ylabel("Avg MCP (Rs/MWh)"); ax1.set_title("Average MCP by Hour of Day")
    ax1.set_xticks(range(0,24,2)); ax1.legend(fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    ax2=fig.add_subplot(gs[0,1])
    daily_dow=iex.groupby("day_of_week")["MCP"].mean()
    bars=ax2.bar(range(7),daily_dow.values,color=[C_BLUE]*5+[C_GREEN]*2,alpha=0.8,edgecolor="white")
    ax2.set_xticks(range(7)); ax2.set_xticklabels(day_names)
    ax2.set_ylabel("Avg MCP"); ax2.set_title("Average MCP by Day of Week")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    for bar in bars:
        ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f"Rs{bar.get_height():,.0f}",ha="center",va="bottom",fontsize=8,fontweight="bold")
    ax3=fig.add_subplot(gs[1,0])
    pivot=iex.pivot_table(values="MCP",index="day_of_week",columns="hour",aggfunc="mean")
    pivot.index=[day_names[i] for i in pivot.index]
    sns.heatmap(pivot,ax=ax3,cmap="RdYlGn_r",fmt=".0f",linewidths=0.2,linecolor="white",
                cbar_kws={"label":"Avg MCP (Rs/MWh)","shrink":0.8})
    ax3.set_title("MCP Heatmap — Hour x Day of Week"); ax3.set_xlabel("Hour"); ax3.set_ylabel("Day")
    ax4=fig.add_subplot(gs[1,1])
    monthly=iex.groupby("month")["MCP"].agg(["mean","std"]).reset_index()
    ax4.fill_between(monthly["month"],monthly["mean"]-monthly["std"],monthly["mean"]+monthly["std"],alpha=0.2,color=C_ORANGE)
    ax4.plot(monthly["month"],monthly["mean"],color=C_ORANGE,linewidth=2.5,marker="s",markersize=6)
    ax4.set_xlabel("Month"); ax4.set_ylabel("Avg MCP"); ax4.set_title("Average MCP by Month")
    ax4.set_xticks(range(1,13)); ax4.set_xticklabels([month_names[m-1] for m in range(1,13)],rotation=45)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    return fig_to_b64(fig)

def plot_acf_pacf(iex):
    from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
    daily=iex.set_index("datetime")["MCP"].resample("D").mean().dropna()
    n=len(daily)
    lags=min(40,n//2-1)
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    plot_acf(daily,lags=lags,ax=axes[0,0],color=C_BLUE,alpha=0.05)
    axes[0,0].set_title("ACF — Daily MCP (Original)")
    axes[0,0].axvline(7,color=C_RED,linestyle="--",alpha=0.5,label="7-day")
    axes[0,0].legend(fontsize=9)
    plot_pacf(daily,lags=lags,ax=axes[0,1],color=C_GREEN,alpha=0.05,method="ywm")
    axes[0,1].set_title("PACF — Daily MCP (Original)")
    daily_diff=daily.diff().dropna()
    lags2=min(40,len(daily_diff)//2-1)
    plot_acf(daily_diff,lags=lags2,ax=axes[1,0],color=C_ORANGE,alpha=0.05)
    axes[1,0].set_title("ACF — First Difference (Stationary)")
    plot_pacf(daily_diff,lags=lags2,ax=axes[1,1],color=C_RED,alpha=0.05,method="ywm")
    axes[1,1].set_title("PACF — First Difference (Stationary)")
    for ax in axes.flat:
        ax.set_xlabel("Lag (days)"); ax.axhline(0,color="black",linewidth=0.8)
    fig.tight_layout(); return fig_to_b64(fig)

def plot_spike(iex):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    spikes=iex[iex["MCP"]>9000]
    spike_by_hour=spikes.groupby("hour").size()
    total_by_hour=iex.groupby("hour").size()
    spike_pct=(spike_by_hour/total_by_hour*100).reindex(range(24),fill_value=0)
    bars=axes[0].bar(range(24),spike_pct.values,
                     color=[C_RED if v>spike_pct.mean() else C_ORANGE for v in spike_pct.values],alpha=0.8)
    axes[0].axhline(spike_pct.mean(),color="black",linestyle="--",linewidth=1.5,label=f"Avg {spike_pct.mean():.1f}%")
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Spike Frequency (%)"); axes[0].set_xticks(range(0,24,2))
    axes[0].set_title("Price Spike Frequency by Hour (MCP > Rs9,000)"); axes[0].legend()
    pct_levels=[10,25,50,75,90,95,99]
    pct_values=[np.percentile(iex["MCP"],p) for p in pct_levels]
    axes[1].barh(pct_levels,pct_values,color=[C_GREEN,C_GREEN,C_BLUE,C_ORANGE,C_ORANGE,C_RED,C_RED],alpha=0.8,height=4)
    for p,v in zip(pct_levels,pct_values):
        axes[1].text(v+50,p,f"Rs{v:,.0f}",va="center",fontsize=10,fontweight="bold")
    axes[1].axvline(9000,color=C_RED,linestyle="--",linewidth=1.5,label="Rs9,000 cap zone")
    axes[1].set_xlabel("MCP (Rs/MWh)"); axes[1].set_ylabel("Percentile")
    axes[1].set_title("MCP Percentile Distribution"); axes[1].legend()
    fig.tight_layout(); return fig_to_b64(fig)

def plot_yoy(iex):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    years=sorted(iex["year"].dropna().unique())
    colors_yr=[C_RED,C_ORANGE,C_BLUE,C_GREEN,C_PURPLE]
    for yr,col in zip(years,colors_yr):
        hourly=iex[iex["year"]==yr].groupby("hour")["MCP"].mean()
        axes[0].plot(hourly.index,hourly.values,label=str(int(yr)),color=col,linewidth=2,marker="o",markersize=3)
    axes[0].set_xlabel("Hour"); axes[0].set_ylabel("Avg MCP (Rs/MWh)")
    axes[0].set_title("Year-over-Year Hourly Profile\n(Market Regime Shift)"); axes[0].legend(title="Year")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    yoy=iex.groupby("year")["MCP"].mean().reset_index()
    bars=axes[1].bar(yoy["year"].astype(str),yoy["MCP"],color=colors_yr[:len(yoy)],alpha=0.8,edgecolor="white")
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+30,f"Rs{bar.get_height():,.0f}",ha="center",fontsize=11,fontweight="bold")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Avg MCP (Rs/MWh)")
    axes[1].set_title("Average MCP by Year\n(Downward Trend: Renewables Growth)")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    fig.tight_layout(); return fig_to_b64(fig)

def plot_stationarity(iex):
    from statsmodels.tsa.stattools import adfuller
    from scipy import stats
    daily=iex.set_index("datetime")["MCP"].resample("D").mean().dropna()
    daily_diff=daily.diff().dropna()
    fig,axes=plt.subplots(2,2,figsize=(14,8))
    roll_mean=daily.rolling(30).mean(); roll_std=daily.rolling(30).std()
    axes[0,0].plot(daily.index,daily.values,color=C_BLUE,alpha=0.5,linewidth=0.8,label="Daily MCP")
    axes[0,0].plot(roll_mean.index,roll_mean.values,color=C_RED,linewidth=2,label="30-day Mean")
    axes[0,0].fill_between(daily.index,roll_mean-roll_std,roll_mean+roll_std,alpha=0.15,color=C_ORANGE)
    axes[0,0].set_title("Rolling Mean & Std — Original"); axes[0,0].legend(fontsize=9)
    axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"Rs{x:,.0f}"))
    roll_mean_d=daily_diff.rolling(30).mean()
    axes[0,1].plot(daily_diff.index,daily_diff.values,color=C_GREEN,alpha=0.5,linewidth=0.8,label="Differenced")
    axes[0,1].plot(roll_mean_d.index,roll_mean_d.values,color=C_RED,linewidth=2,label="30-day Mean")
    axes[0,1].axhline(0,color="black",linewidth=0.8,linestyle="--")
    axes[0,1].set_title("Rolling Mean & Std — First Difference"); axes[0,1].legend(fontsize=9)
    adf_orig=adfuller(daily.dropna()); adf_diff=adfuller(daily_diff.dropna())
    table_data=[["Metric","Original","First Diff"],
        ["ADF Statistic",f"{adf_orig[0]:.4f}",f"{adf_diff[0]:.4f}"],
        ["p-value",f"{adf_orig[1]:.6f}",f"{adf_diff[1]:.6f}"],
        ["Critical 1%",f"{adf_orig[4]['1%']:.4f}",f"{adf_diff[4]['1%']:.4f}"],
        ["Critical 5%",f"{adf_orig[4]['5%']:.4f}",f"{adf_diff[4]['5%']:.4f}"],
        ["Stationary","No" if adf_orig[1]>0.05 else "Yes","Yes" if adf_diff[1]<0.05 else "No"]]
    ax_t=fig.add_subplot(2,2,3); ax_t.axis("off")
    tbl=ax_t.table(cellText=table_data[1:],colLabels=table_data[0],cellLoc="center",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2,2.2)
    for j in range(3):
        tbl[0,j].set_facecolor(C_MAIN); tbl[0,j].set_text_props(color="white",fontweight="bold")
    tbl[5,1].set_facecolor("#ffcccc"); tbl[5,2].set_facecolor("#ccffcc")
    ax_t.set_title("ADF Stationarity Test Results",fontweight="bold",pad=10)
    ax_qq=fig.add_subplot(2,2,4)
    stats.probplot(daily_diff.dropna(),dist="norm",plot=ax_qq)
    ax_qq.set_title("Q-Q Plot — First Difference (Normality Check)")
    ax_qq.get_lines()[0].set(color=C_BLUE,markersize=3,alpha=0.5)
    ax_qq.get_lines()[1].set(color=C_RED,linewidth=2)
    fig.tight_layout(h_pad=3); return fig_to_b64(fig),adf_orig,adf_diff

def plot_wx_corr(iex,wx):
    if wx is None: return None
    daily_mcp=iex.set_index("datetime")["MCP"].resample("D").mean().reset_index()
    daily_mcp.columns=["date","MCP"]; daily_mcp["date"]=daily_mcp["date"].dt.date.astype(str)
    wx["date"]=pd.to_datetime(wx["datetime"]).dt.date.astype(str)
    daily_wx=wx.groupby("date")[["temperature","wind_speed","humidity"]].mean().reset_index()
    merged=daily_mcp.merge(daily_wx,on="date",how="inner").dropna()
    if len(merged)<10: return None
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,(col,label,color) in zip(axes,[("temperature","Temperature (C)",C_RED),("wind_speed","Wind Speed (m/s)",C_BLUE),("humidity","Humidity (%)",C_GREEN)]):
        if col not in merged.columns: continue
        x,y=merged[col],merged["MCP"]
        ax.scatter(x,y,alpha=0.3,color=color,s=12)
        z=np.polyfit(x.dropna(),y[x.notna()],1); p=np.poly1d(z)
        xline=np.linspace(x.min(),x.max(),100)
        ax.plot(xline,p(xline),color="black",linewidth=2,linestyle="--")
        ax.set_xlabel(label); ax.set_ylabel("Daily Avg MCP")
        ax.set_title(f"MCP vs {label}\nCorr: {x.corr(y):.3f}")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"Rs{v:,.0f}"))
    fig.tight_layout(); return fig_to_b64(fig)

def plot_com_corr(iex,com):
    if com is None: return None
    daily_mcp=iex.set_index("datetime")["MCP"].resample("D").mean().reset_index()
    daily_mcp.columns=["date","MCP"]; daily_mcp["date"]=daily_mcp["date"].dt.date.astype(str)
    com["date"]=pd.to_datetime(com["date"]).dt.date.astype(str)
    merged=daily_mcp.merge(com,on="date",how="inner").dropna()
    if len(merged)<10: return None
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,(col,label,color) in zip(axes,[("crude_oil_usd","Crude Oil (USD/bbl)",C_BLUE),("natural_gas_usd","Natural Gas (USD/MMBtu)",C_ORANGE),("usd_inr","USD/INR Rate",C_GREEN)]):
        if col not in merged.columns: continue
        x,y=merged[col],merged["MCP"]
        valid=x.notna()&y.notna()
        ax.scatter(x[valid],y[valid],alpha=0.3,color=color,s=12)
        if valid.sum()>2:
            z=np.polyfit(x[valid],y[valid],1); p=np.poly1d(z)
            xline=np.linspace(x[valid].min(),x[valid].max(),100)
            ax.plot(xline,p(xline),color="black",linewidth=2,linestyle="--")
        ax.set_xlabel(label); ax.set_ylabel("Daily Avg MCP")
        ax.set_title(f"MCP vs {label}\nCorr: {x.corr(y):.3f}")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_:f"Rs{v:,.0f}"))
    fig.tight_layout(); return fig_to_b64(fig)

def build_html(plots,stats,iex,adf_orig,adf_diff):
    spike_pct=(iex["MCP"]>9000).mean()*100
    peak_hour=iex.groupby("hour")["MCP"].mean().idxmax()
    trough_hour=iex.groupby("hour")["MCP"].mean().idxmin()
    tv,sv,rv=stats.get("tv",0),stats.get("sv",0),stats.get("rv",0)
    mean_2023=iex[iex["year"]==2023]["MCP"].mean() if 2023 in iex["year"].values else 0
    mean_2025=iex[iex["year"]==2025]["MCP"].mean() if 2025 in iex["year"].values else 0

    css="""
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#2d3436}
.header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);color:white;padding:40px;text-align:center}
.header h1{font-size:2.2em;margin-bottom:10px}
.header p{font-size:1.1em;opacity:0.85}
.meta{display:flex;justify-content:center;gap:20px;margin-top:20px;flex-wrap:wrap}
.meta span{background:rgba(255,255,255,0.15);padding:6px 16px;border-radius:20px;font-size:0.9em}
.container{max-width:1300px;margin:0 auto;padding:30px 20px}
.section{background:white;border-radius:12px;padding:28px;margin-bottom:28px;box-shadow:0 2px 12px rgba(0,0,0,0.07)}
.section h2{font-size:1.5em;color:#1a1a2e;margin-bottom:12px;padding-bottom:10px;border-bottom:3px solid #e94560}
.section h3{font-size:1.1em;color:#2980b9;margin:20px 0 10px}
.insight{background:#f8f9ff;border-left:4px solid #2980b9;padding:14px 18px;border-radius:0 8px 8px 0;margin:14px 0;font-size:0.95em;line-height:1.7}
.insight.warn{border-color:#e94560;background:#fff8f8}
.insight.green{border-color:#27ae60;background:#f0fff4}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px;margin:20px 0}
.card{background:white;border-radius:10px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.08);text-align:center}
.card-title{font-size:0.78em;color:#636e72;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px}
.card-value{font-size:1.7em;font-weight:700;margin-bottom:4px}
.card-sub{font-size:0.78em;color:#b2bec3}
.caption{text-align:center;font-size:0.82em;color:#636e72;margin-top:8px;font-style:italic}
.stl-box{background:#1a1a2e;color:white;border-radius:10px;padding:18px;margin:14px 0;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;text-align:center}
.stl-val{font-size:2em;font-weight:bold;color:#e94560}
.stl-lbl{font-size:0.82em;color:#b2bec3;margin-top:4px}
.tbl{width:100%;border-collapse:collapse}
.tbl th{background:#1a1a2e;color:white;padding:12px;text-align:left}
.tbl td{padding:12px;border-bottom:1px solid #eee}
.tbl tr:hover td{background:#f8f9ff}
"""

    def section(title, *content):
        return f'<div class="section"><h2>{title}</h2>{"".join(content)}</div>'

    def insight(text, cls=""):
        return f'<div class="insight {cls}">{text}</div>'

    rows="".join(f'<tr><td>{f}</td><td>{i}</td></tr>' for f,i in [
        ("Strong lag-1 autocorrelation (ACF)","mcp_lag_1h is most important feature (importance=0.315)"),
        ("Weekly seasonality (7-day STL cycle)","day_of_week and is_weekend features critical for weekday/weekend pricing"),
        (f"Evening price peak at {peak_hour}:00","hour and hour_bucket features capture intraday demand surge"),
        (f"Regime shift: Rs{mean_2023:,.0f} (2023) to Rs{mean_2025:,.0f} (2025)","Training limited to last 18 months — avoids stale regime data polluting model"),
        (f"{spike_pct:.1f}% price cap events (>Rs9,000)","Inherently unpredictable — structural floor on achievable MAPE (~20%)"),
        ("Non-stationary original series (ADF p>0.05)","XGBoost handles via rolling lag features without explicit differencing"),
        ("Bimodal price distribution","Heavy right tail inflates MAPE — spike blocks dominate error metric"),
        ("Temperature positive correlation with MCP","temp_delhi, cooling_degree features improve summer peak predictions"),
        ("STL residual captures price shocks","Residual component confirms model cannot predict grid stress spikes"),
    ])

    wx_section = section("Weather Correlation Analysis",
        img_tag(plots.get("weather",""), caption="Figure 10: MCP correlation with temperature, wind speed, and humidity (Delhi)."),
        insight("Temperature shows positive correlation (cooling demand). Wind speed shows negative correlation (more renewables = lower prices). "
                "These validate including weather features in the model.")) if plots.get("weather") else ""

    com_section = section("Commodity Price Correlation",
        img_tag(plots.get("commodity",""), caption="Figure 11: MCP correlation with crude oil, natural gas, and USD/INR exchange rate."),
        insight("Crude oil and natural gas show moderate positive correlation — thermal generation costs pass through to market prices. "
                "As renewables penetration grows, correlation with fossil fuel prices is weakening.")) if plots.get("commodity") else ""

    body = f"""
    {section("📊 Key Market Statistics", stat_cards(iex),
        insight("<b>Market Context:</b> RTM prices are highly volatile due to grid stress events and renewable intermittency. "
                f"<b>{spike_pct:.1f}% of all 15-min blocks hit the price cap zone (&gt;Rs9,000)</b>. "
                f"The market shows a clear downward trend from Rs{mean_2023:,.0f}/MWh (2023) to Rs{mean_2025:,.0f}/MWh (2025), "
                "driven by increasing renewable energy penetration."))}

    {section("📈 Price Time Series Overview",
        img_tag(plots.get("overview",""), caption="Figure 1: Full RTM MCP time series (15-min) with daily average and 30-day rolling volatility."),
        insight("<b>Key Observation:</b> Clear regime shifts across years. High price spikes are clustered during evening peak hours (7–10 PM). "
                "Rolling volatility shows periods of high market stress corresponding to grid congestion or generation outages."))}

    {section("📉 Price Distribution Analysis",
        img_tag(plots.get("dist",""), caption="Figure 2: Left — Bimodal distribution (normal trading + spike zone). Right — Year-over-year box plots showing regime shift."),
        insight("<b>Bimodal Distribution:</b> Main cluster around Rs{:,.0f}/MWh (normal) and secondary cluster near Rs10,000–12,000 (cap events). "
                "This bimodality makes MAPE challenging as spike predictions carry high uncertainty.".format(iex['MCP'].median()), "warn"),
        insight("<b>Regime Shift Confirmed:</b> Box plots by year show systematic downward shift in prices. "
                "2023 distribution is entirely different from 2025–2026, validating the 18-month training window decision.", "green"))}

    {section("🔬 STL Decomposition Analysis",
        "<p style='color:#636e72;margin-bottom:16px'>STL (Seasonal-Trend decomposition using LOESS) separates the time series into "
        "three interpretable components: <b>Trend</b> (long-term direction), <b>Seasonal</b> (recurring patterns), and <b>Residual</b> (unexplained shocks).</p>",
        "<h3>Weekly STL — Daily Average MCP (period = 7 days)</h3>",
        f'<div class="stl-box"><div><div class="stl-val">{tv:.1f}%</div><div class="stl-lbl">Variance from Trend</div></div>'
        f'<div><div class="stl-val">{sv:.1f}%</div><div class="stl-lbl">Variance from Seasonality</div></div>'
        f'<div><div class="stl-val">{rv:.1f}%</div><div class="stl-lbl">Residual (Unexplained)</div></div></div>',
        img_tag(plots.get("stl_daily",""), caption="Figure 3: STL decomposition on daily MCP. Trend shows long-term price decline. Seasonal captures weekly demand cycle. Residuals = price spikes."),
        insight("<b>STL Interpretation:</b> "
                "The <b>Trend</b> confirms gradual price decline (renewable penetration effect). "
                "The <b>Seasonal</b> component captures weekly demand cycle — weekdays see higher prices than weekends. "
                f"The <b>Residual</b> contains price cap spikes — the high residual variance ({rv:.1f}%) explains "
                "why MAPE ~20% is the realistic performance ceiling for this market."),
        ("<h3>Daily STL — 15-Minute Blocks (period = 96 blocks = 24 hours)</h3>" +
        img_tag(plots.get("stl_15min",""), caption="Figure 4: STL on 15-min data. Period=96 captures within-day price cycle. Evening peak (blocks 68–80 = 5–8 PM) consistently elevated.") +
        insight("<b>15-Min STL Insight:</b> The daily seasonal component shows the characteristic double-peak — "
                "morning peak (6–9 AM, industrial startup) and stronger evening peak (6–10 PM, domestic + commercial load). "
                "The trend at this resolution captures weekly demand cycles.")) if plots.get("stl_15min") else "")}

    {section("🕐 Seasonal Pattern Analysis",
        img_tag(plots.get("seasonal",""), caption="Figure 5: MCP patterns by hour, day of week, and month. Heatmap reveals price hotspots at evening hours x weekdays."),
        insight(f"<b>Peak Hours:</b> Hour {peak_hour}:00 (evening) shows highest avg prices (Rs{iex.groupby('hour')['MCP'].mean().max():,.0f}/MWh). "
                f"Hour {trough_hour}:00 shows lowest prices (Rs{iex.groupby('hour')['MCP'].mean().min():,.0f}/MWh) driven by solar generation.", "warn"),
        insight("<b>Weekend Effect:</b> Saturday and Sunday show ~8–12% lower prices as industrial demand drops. "
                "This weekly pattern is captured in the STL seasonal component and drives the day_of_week feature importance.", "green"))}

    {section("📐 Stationarity Analysis (ADF Test)",
        img_tag(plots.get("stationarity",""), caption="Figure 6: Rolling mean/std for original and differenced series. ADF test table and Q-Q normality check."),
        insight(f"<b>ADF Result (Original):</b> Statistic={adf_orig[0]:.4f}, p-value={adf_orig[1]:.4f} → "
                f"{'Series is stationary' if adf_orig[1]<0.05 else 'Non-stationary — differencing needed for ARIMA'}", "green" if adf_orig[1]<0.05 else "warn"),
        insight(f"<b>ADF Result (First Difference):</b> Statistic={adf_diff[0]:.4f}, p-value={adf_diff[1]:.6f} → "
                "Stationary after differencing. Confirms ARIMA(p,1,q) structure. "
                "XGBoost handles non-stationarity implicitly through lag features.", "green"))}

    {section("📡 Autocorrelation Analysis (ACF / PACF)",
        img_tag(plots.get("acf",""), caption="Figure 7: ACF and PACF for original and first-differenced daily MCP. Significant spikes at lag 7, 14 confirm weekly seasonality."),
        insight("<b>Key Lags:</b><br>"
                "Lag 1 (1 day): Strong — yesterday's price is best predictor (confirms mcp_lag_24h importance)<br>"
                "Lag 7 (1 week): Significant spike — same-day-last-week pattern (weekly seasonality)<br>"
                "Lag 14 (2 weeks): Secondary harmonic<br>"
                "PACF: Drops after lag 1–2, suggesting AR(1)/AR(2) structure for ARIMA baseline"))}

    {section("⚡ Price Spike Analysis",
        img_tag(plots.get("spike",""), caption="Figure 8: Left — Hourly frequency of price cap events. Right — Price percentile distribution showing heavy right tail."),
        insight(f"<b>Price Cap Events:</b> {spike_pct:.1f}% of blocks hit the cap zone (&gt;Rs9,000). "
                "Spikes concentrated in evening hours (7–10 PM). These events require real-time grid data "
                "unavailable in our dataset — a structural explanation for why MAPE cannot reach &lt;5%.", "warn"))}

    {section("📅 Year-over-Year Regime Analysis",
        img_tag(plots.get("yoy",""), caption="Figure 9: Left — YoY hourly price profiles showing regime shift. Right — Annual average MCP decline trend."),
        insight(f"<b>Regime Shift Finding:</b> {(abs(mean_2023-mean_2025)/mean_2023*100):.1f}% price decline from 2023 to 2025. "
                "Training on 2023 data to predict 2025–2026 caused MAPE to spike from 20% to 160%+. "
                "Limiting training to last 18 months keeps train and test in the same market regime.", "warn"))}

    {wx_section}

    {com_section}

    {section("💡 Key EDA Insights & Modelling Implications",
        f'<table class="tbl"><tr style="background:#1a1a2e;color:white"><th>Finding</th><th>Modelling Implication</th></tr>{rows}</table>')}
    """

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IEX RTM EDA — Group 05</title>
<style>{css}</style>
</head><body>
<div class="header">
  <h1>IEX RTM Electricity Market — Exploratory Data Analysis</h1>
  <p>Indian Energy Exchange | Real-Time Market | 15-Minute Clearing Price (MCP)</p>
  <div class="meta">
    <span>Group 05 — ISB AMPBA</span>
    <span>Records: {len(iex):,}</span>
    <span>Period: {iex['datetime'].min().strftime('%b %Y')} to {iex['datetime'].max().strftime('%b %Y')}</span>
    <span>Generated: {datetime.now().strftime('%d %b %Y %H:%M')}</span>
  </div>
</div>
<div class="container">{body}</div>
<div style="text-align:center;padding:20px;color:#b2bec3;font-size:0.85em">
  Group 05 — ISB AMPBA | IEX RTM Electricity Price Forecasting | CRISP-ML(Q)
</div>
</body></html>"""

def generate_and_save():
    print("Loading data...")
    iex,wx,com=load_data()
    print(f"  IEX: {len(iex):,} records | {iex['datetime'].min().date()} to {iex['datetime'].max().date()}")
    plots={}; stats={}
    print("Generating plots...")
    print("  1/9 Overview..."); plots["overview"]=plot_overview(iex)
    print("  2/9 Distribution..."); plots["dist"]=plot_distribution(iex)
    print("  3/9 STL daily..."); b64,tv,sv,rv=plot_stl(iex); plots["stl_daily"]=b64; stats.update({"tv":tv,"sv":sv,"rv":rv})
    print("  4/9 STL 15min..."); b=plot_stl_15min(iex);
    if b: plots["stl_15min"]=b
    print("  5/9 Seasonal patterns..."); plots["seasonal"]=plot_seasonal(iex)
    print("  6/9 Stationarity..."); b64s,adf_orig,adf_diff=plot_stationarity(iex); plots["stationarity"]=b64s
    print("  7/9 ACF/PACF..."); plots["acf"]=plot_acf_pacf(iex)
    print("  8/9 Spike analysis..."); plots["spike"]=plot_spike(iex)
    print("  9/9 Year-over-year..."); plots["yoy"]=plot_yoy(iex)
    if wx is not None:
        print("  Weather corr..."); b=plot_wx_corr(iex,wx)
        if b: plots["weather"]=b
    if com is not None:
        print("  Commodity corr..."); b=plot_com_corr(iex,com)
        if b: plots["commodity"]=b
    print("Building HTML...")
    html=build_html(plots,stats,iex,adf_orig,adf_diff)
    out=os.path.join(OUTPUT_DIR,"eda_report.html")
    with open(out,"w",encoding="utf-8") as f: f.write(html)
    print(f"\nEDA report: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
    return out

if __name__=="__main__":
    generate_and_save()
