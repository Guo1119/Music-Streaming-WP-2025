# EDA Impressions Dataset
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Establish connection
conn = duckdb.connect('/Users/michaelmlr/Desktop/Master/Data/ncm-research-data.duckdb')

## 1. Basic Information 

print("="*60)
print("1. TABLE STRUCTURE")
print("="*60)

# Show schema
schema = conn.execute("DESCRIBE impression").fetchdf()
print("\nColumns and Data Types:")
print(schema)

# Number of rows and columns
row_count = conn.execute("SELECT COUNT(*) FROM impression").fetchone()[0]
col_count = len(schema)
print(f"\nNumber of Rows: {row_count:,}")
print(f"Number of Columns: {col_count}")
## Preview first rows
print("\n" + "="*60)
print("2. FIRST ROWS")
print("="*60)
df_sample = conn.execute("SELECT * FROM impression LIMIT 5").fetchdf()
print(df_sample)
## Descriptive statistics 
print("\n" + "="*60)
print("3. DESCRIPTIVE STATISTICS")
print("="*60)

# For numeric columns
numeric_stats = conn.execute("""
    SELECT * FROM impression LIMIT 10000
""").fetchdf().describe()
print(numeric_stats)
## Missing Values 
print("\n" + "="*60)
print("4. MISSING VALUES")
print("="*60)

# Check for NULL values per column
for col in schema['column_name']:
    null_count = conn.execute(f"""
        SELECT COUNT(*) 
        FROM impression 
        WHERE {col} IS NULL
    """).fetchone()[0]
    null_percent = (null_count / row_count) * 100
    if null_count > 0:
        print(f"{col}: {null_count:,} ({null_percent:.2f}%)")
## Unique Values 

print("\n" + "="*60)
print("5. UNIQUE VALUES PER COLUMN")
print("="*60)

for col in schema['column_name']:
    unique_count = conn.execute(f"""
        SELECT COUNT(DISTINCT {col}) 
        FROM impression
    """).fetchone()[0]
    print(f"{col}: {unique_count:,} unique values")
# 2. Visualisations 
## Settings for visualization
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Establish connection
conn = duckdb.connect('/Users/michaelmlr/Desktop/Master/Data/ncm-research-data.duckdb')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

# ===== LOAD DATA FOR VISUALIZATION =====
# Load a sample (100k rows for performance)
df = conn.execute("SELECT * FROM impression LIMIT 100000").fetchdf()

# Convert timestamp to datetime
df['impressTime_dt'] = pd.to_datetime(df['impressTime'], unit='ms')
df['date'] = df['impressTime_dt'].dt.date

# ===== CREATE VISUALIZATIONS =====
fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 1. Distribution of dt (days)
axes[0, 0].hist(df['dt'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribution of Days (dt)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Days')
axes[0, 0].set_ylabel('Frequency')

# 2. Distribution of impressPosition
axes[0, 1].hist(df['impressPosition'], bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].set_title('Distribution of Impression Position', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Position')
axes[0, 1].set_ylabel('Frequency')

# 3. Click Rate (isClick)
click_counts = df['isClick'].value_counts()
axes[0, 2].bar(['No Click', 'Click'], click_counts.values, color=['lightcoral', 'lightgreen'])
axes[0, 2].set_title(f'Click Rate: {(click_counts[1]/len(df)*100):.2f}%', fontsize=12, fontweight='bold')
axes[0, 2].set_ylabel('Count')

# 4. Engagement Actions Comparison
engagement_data = {
    'Click': df['isClick'].sum(),
    'Comment': df['isComment'].sum(),
    'Share': df['isShare'].sum(),
    'Like': df['isLike'].sum(),
    'View Comment': df['isViewComment'].sum(),
    'Into Homepage': df['isIntoPersonalHomepage'].sum()
}
axes[1, 0].bar(engagement_data.keys(), engagement_data.values(), color='skyblue')
axes[1, 0].set_title('Engagement Actions Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Impressions over Time
daily_impressions = df.groupby('date').size()
axes[1, 1].plot(daily_impressions.index, daily_impressions.values, linewidth=2, color='purple')
axes[1, 1].set_title('Impressions Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Number of Impressions')
axes[1, 1].tick_params(axis='x', rotation=45)

# 6. Click Rate by Position (binned)
df['position_bin'] = pd.cut(df['impressPosition'], bins=10)
click_by_position = df.groupby('position_bin')['isClick'].agg(['sum', 'count'])
click_by_position['rate'] = click_by_position['sum'] / click_by_position['count'] * 100
axes[1, 2].bar(range(len(click_by_position)), click_by_position['rate'], color='teal')
axes[1, 2].set_title('Click Rate by Position', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('Position Bin')
axes[1, 2].set_ylabel('Click Rate (%)')

# 7. Distribution of mlogViewTime (for non-null values)
df_view = df[df['mlogViewTime'].notna()]
if len(df_view) > 0:
    axes[2, 0].hist(df_view['mlogViewTime'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[2, 0].set_title('Distribution of Mlog View Time (seconds)', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('View Time (seconds)')
    axes[2, 0].set_ylabel('Frequency')

# 8. Engagement Rate by Day
engagement_by_day = df.groupby('dt').agg({
    'isClick': 'mean',
    'isLike': 'mean',
    'isComment': 'mean',
    'isShare': 'mean'
}) * 100
axes[2, 1].plot(engagement_by_day.index, engagement_by_day['isClick'], label='Click', marker='o')
axes[2, 1].plot(engagement_by_day.index, engagement_by_day['isLike'], label='Like', marker='s')
axes[2, 1].plot(engagement_by_day.index, engagement_by_day['isComment'], label='Comment', marker='^')
axes[2, 1].plot(engagement_by_day.index, engagement_by_day['isShare'], label='Share', marker='d')
axes[2, 1].set_title('Engagement Rate by Day', fontsize=12, fontweight='bold')
axes[2, 1].set_xlabel('Day (dt)')
axes[2, 1].set_ylabel('Rate (%)')
axes[2, 1].legend()

# 9. Hourly Impressions Pattern
df['hour'] = df['impressTime_dt'].dt.hour
hourly_impressions = df.groupby('hour').size()
axes[2, 2].bar(hourly_impressions.index, hourly_impressions.values, color='gold')
axes[2, 2].set_title('Impressions by Hour of Day', fontsize=12, fontweight='bold')
axes[2, 2].set_xlabel('Hour')
axes[2, 2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('impression_eda_visualizations.png', dpi=300, bbox_inches='tight')
plt.show()
# ===== ADDITIONAL INSIGHTS =====
print("\n" + "="*60)
print("KEY INSIGHTS")
print("="*60)

print(f"\n1. CLICK RATE: {(df['isClick'].sum()/len(df)*100):.3f}%")
print(f"2. LIKE RATE: {(df['isLike'].sum()/len(df)*100):.3f}%")
print(f"3. COMMENT RATE: {(df['isComment'].sum()/len(df)*100):.3f}%")
print(f"4. SHARE RATE: {(df['isShare'].sum()/len(df)*100):.3f}%")

print(f"\n5. AVERAGE POSITION: {df['impressPosition'].mean():.2f}")
print(f"6. MEDIAN POSITION: {df['impressPosition'].median():.2f}")

print(f"\n7. UNIQUE USERS: {df['userId'].nunique():,}")
print(f"8. UNIQUE MLOGS: {df['mlogId'].nunique():,}")
print(f"9. AVERAGE IMPRESSIONS PER USER: {len(df)/df['userId'].nunique():.2f}")

conn.close()

print("\n✅ Visualizations saved as 'impression_eda_visualizations.png'")
