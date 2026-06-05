import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 130

OUT = '../outputs/eda_plots'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv('../data/raw/retail_store_inventory.csv', parse_dates=['Date'])
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')
df['day_of_week'] = df['Date'].dt.day_name()
df['month'] = df['Date'].dt.month
df['stockout'] = (df['Inventory_Level'] == 0).astype(int)

print(f"Shape          : {df.shape}")
print(f"Date range     : {df.Date.min().date()} → {df.Date.max().date()}")
print(f"Unique Stores  : {df.Store_ID.nunique()}")
print(f"Unique SKUs    : {df.Product_ID.nunique()}")
print(f"SKU×Store combos: {df.groupby(['Store_ID','Product_ID']).ngroups:,}")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Zero-demand days: {(df['Units_Sold']==0).sum():,} ({(df['Units_Sold']==0).mean()*100:.1f}%)")
print(f"Overall stockout rate: {df['stockout'].mean()*100:.2f}%\n")

# ── 1. Demand Distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(df['Units_Sold'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Units Sold', fontsize=13)
axes[0].set_xlabel('Units Sold'); axes[0].set_ylabel('Frequency')
axes[1].hist(np.log1p(df['Units_Sold']), bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('Log Distribution of Units Sold', fontsize=13)
axes[1].set_xlabel('log(Units Sold + 1)')
plt.suptitle('1 · Demand Distribution', fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/01_demand_distribution.png', bbox_inches='tight'); plt.close()
print("✓ 01_demand_distribution.png")

# ── 2. Demand by Category ─────────────────────────────────────────────────────
cat_summary = df.groupby('Category')['Units_Sold'].agg(['mean','median']).sort_values('mean', ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
x = np.arange(len(cat_summary))
ax.bar(x - 0.2, cat_summary['mean'],   0.4, label='Mean',   color='steelblue')
ax.bar(x + 0.2, cat_summary['median'], 0.4, label='Median', color='darkorange')
ax.set_xticks(x); ax.set_xticklabels(cat_summary.index, rotation=30)
ax.set_title('2 · Avg vs Median Demand by Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Units Sold'); ax.legend()
plt.tight_layout(); plt.savefig(f'{OUT}/02_demand_by_category.png', bbox_inches='tight'); plt.close()
print("✓ 02_demand_by_category.png")

# ── 3. Demand Over Time ───────────────────────────────────────────────────────
daily = df.groupby('Date')['Units_Sold'].sum()
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily.index, daily.values, color='steelblue', linewidth=0.7, alpha=0.5, label='Daily')
ax.plot(daily.index, daily.rolling(28).mean(), color='tomato', linewidth=2, label='28-day MA')
ax.set_title('3 · Total Daily Demand Over Time', fontsize=13, fontweight='bold')
ax.set_xlabel('Date'); ax.set_ylabel('Total Units Sold'); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig(f'{OUT}/03_demand_over_time.png', bbox_inches='tight'); plt.close()
print("✓ 03_demand_over_time.png")

# ── 4. Seasonality ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow = df.groupby('day_of_week')['Units_Sold'].mean().reindex(dow_order)
axes[0].bar(range(7), dow.values, color='steelblue')
axes[0].set_xticks(range(7)); axes[0].set_xticklabels(['M','T','W','Th','F','Sa','Su'])
axes[0].set_title('By Day of Week'); axes[0].set_ylabel('Avg Units Sold')

monthly = df.groupby('month')['Units_Sold'].mean()
axes[1].bar(monthly.index, monthly.values, color='darkorange')
axes[1].set_xticks(range(1,13))
axes[1].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[1].set_title('By Month')

season_order = ['Spring','Summer','Autumn','Winter']
season = df.groupby('Seasonality')['Units_Sold'].mean().reindex(season_order)
axes[2].bar(season_order, season.values, color=['#4CAF50','#FF9800','#F44336','#2196F3'])
axes[2].set_title('By Season')

plt.suptitle('4 · Seasonality Patterns', fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/04_seasonality.png', bbox_inches='tight'); plt.close()
print("✓ 04_seasonality.png")

# ── 5. Stockout Analysis ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
stockout_cat = df.groupby('Category')['stockout'].mean().sort_values(ascending=False) * 100
axes[0].bar(stockout_cat.index, stockout_cat.values, color='tomato')
axes[0].set_title('Stockout Rate (%) by Category')
axes[0].set_ylabel('Stockout Rate (%)'); axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(stockout_cat.values):
    axes[0].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=8)

stockout_time = df.groupby('Date')['stockout'].mean() * 100
axes[1].plot(stockout_time.index, stockout_time.rolling(28).mean(), color='tomato', linewidth=2)
axes[1].set_title('28-day Rolling Stockout Rate Over Time')
axes[1].set_ylabel('Stockout Rate (%)'); axes[1].set_xlabel('Date')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('5 · Stockout Analysis', fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/05_stockout_analysis.png', bbox_inches='tight'); plt.close()
print("✓ 05_stockout_analysis.png")

# ── 6. Promotions & Weather ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
promo = df.groupby('Holiday_Promotion')['Units_Sold'].mean()
promo.index = ['No Promo', 'Promo']
bars = axes[0].bar(promo.index, promo.values, color=['steelblue','gold'], width=0.4)
axes[0].set_title('Promotion Impact on Demand')
axes[0].set_ylabel('Avg Units Sold')
for bar, v in zip(bars, promo.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}', ha='center', fontweight='bold')

weather = df.groupby('Weather_Condition')['Units_Sold'].mean().sort_values(ascending=False)
axes[1].bar(weather.index, weather.values, color=['#64B5F6','#81C784','#FFD54F'])
axes[1].set_title('Weather Condition Impact on Demand')
axes[1].set_ylabel('Avg Units Sold')
for i, v in enumerate(weather.values):
    axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontweight='bold')

plt.suptitle('6 · External Demand Drivers', fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/06_promotions_weather.png', bbox_inches='tight'); plt.close()
print("✓ 06_promotions_weather.png")

# ── 7. Correlation Heatmap ────────────────────────────────────────────────────
num_cols = ['Units_Sold','Inventory_Level','Units_Ordered','Demand_Forecast','Price','Discount','Competitor_Pricing']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('7 · Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/07_correlation_heatmap.png', bbox_inches='tight'); plt.close()
print("✓ 07_correlation_heatmap.png")

# ── 8. SKU Deep Dive ──────────────────────────────────────────────────────────
top_skus = df.groupby('Product_ID')['Units_Sold'].sum().nlargest(3).index.tolist()
sample_store = df['Store_ID'].value_counts().index[0]
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for i, sku in enumerate(top_skus):
    sku_df = df[(df['Product_ID'] == sku) & (df['Store_ID'] == sample_store)].sort_values('Date')
    axes[i].plot(sku_df['Date'], sku_df['Units_Sold'], color='steelblue', linewidth=1, label='Actual Demand')
    axes[i].plot(sku_df['Date'], sku_df['Demand_Forecast'], color='tomato', linewidth=1, linestyle='--', label='Baseline Forecast')
    axes[i].fill_between(sku_df['Date'], sku_df['Units_Sold'], sku_df['Demand_Forecast'],
                          where=sku_df['Demand_Forecast'] < sku_df['Units_Sold'],
                          alpha=0.2, color='tomato', label='Forecast Gap (stockout risk)')
    axes[i].set_title(f'SKU: {sku} | Store: {sample_store}', fontsize=10)
    axes[i].set_ylabel('Units Sold'); axes[i].legend(fontsize=8)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.suptitle('8 · Top SKU Deep Dive: Actual vs Baseline Forecast', fontweight='bold', fontsize=13)
plt.tight_layout(); plt.savefig(f'{OUT}/08_sku_deep_dive.png', bbox_inches='tight'); plt.close()
print("✓ 08_sku_deep_dive.png")

print(f"\n✅ All plots saved to {OUT}/")
