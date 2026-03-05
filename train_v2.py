import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              classification_report, confusion_matrix, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & FEATURE ENGINEERING
# ─────────────────────────────────────────────
df = pd.read_csv('indian_employment_dataset_2016_2024.csv')
df = df.sort_values(['Sector','Year','Month']).reset_index(drop=True)

le = LabelEncoder()
df['Sector_enc'] = le.fit_transform(df['Sector'])

# Time index
df['t'] = (df['Year'] - 2016) * 12 + df['Month']

# Lag features per sector
for col in ['LFPR','UR','WPR','GDP_Growth_Rate']:
    df[f'{col}_lag1'] = df.groupby('Sector')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('Sector')[col].shift(2)
    df[f'{col}_lag3'] = df.groupby('Sector')[col].shift(3)
    df[f'{col}_lag6'] = df.groupby('Sector')[col].shift(6)
    df[f'{col}_lag12'] = df.groupby('Sector')[col].shift(12)

# Rolling stats (using only past values)
for col in ['LFPR','UR','WPR']:
    df[f'{col}_roll3']  = df.groupby('Sector')[col].transform(lambda x: x.shift(1).rolling(3).mean())
    df[f'{col}_roll6']  = df.groupby('Sector')[col].transform(lambda x: x.shift(1).rolling(6).mean())
    df[f'{col}_roll3_std'] = df.groupby('Sector')[col].transform(lambda x: x.shift(1).rolling(3).std())

# Trend: month-over-month change
df['UR_diff1']   = df.groupby('Sector')['UR'].diff(1)
df['LFPR_diff1'] = df.groupby('Sector')['LFPR'].diff(1)

# Seasonality
df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# Target: NEXT month's UR
df['UR_next'] = df.groupby('Sector')['UR'].shift(-1)

# Classification: UR > 65th percentile = stress
ur_thresh = df['UR'].quantile(0.65)
df['UR_stress_next'] = (df['UR_next'] > ur_thresh).astype(int)

df_clean = df.dropna().reset_index(drop=True)
print(f"Clean rows: {len(df_clean)}  |  UR threshold: {ur_thresh:.2f}%")

FEATURES = [
    'Sector_enc', 'Month', 'month_sin', 'month_cos', 't',
    'GDP_Growth_Rate', 'GDP_Growth_Rate_lag1',
    'LFPR_lag1','LFPR_lag2','LFPR_lag3','LFPR_lag6','LFPR_lag12',
    'UR_lag1', 'UR_lag2', 'UR_lag3', 'UR_lag6', 'UR_lag12',
    'WPR_lag1', 'WPR_lag2', 'WPR_lag3',
    'UR_roll3','UR_roll6','UR_roll3_std',
    'LFPR_roll3','WPR_roll3',
    'UR_diff1','LFPR_diff1',
]

X     = df_clean[FEATURES]
y_reg = df_clean['UR_next']
y_clf = df_clean['UR_stress_next']

# Time-aware split: last 18 months = test
split = len(df_clean) - 90  # 18 months × 5 sectors
X_train, X_test = X.iloc[:split], X.iloc[split:]
yr_train, yr_test = y_reg.iloc[:split], y_reg.iloc[split:]
yc_train, yc_test = y_clf.iloc[:split], y_clf.iloc[split:]

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 2. RANDOM FOREST REGRESSOR
# ─────────────────────────────────────────────
rf = RandomForestRegressor(n_estimators=300, max_depth=8,
                           min_samples_leaf=4, max_features=0.7,
                           random_state=42, n_jobs=-1)
rf.fit(X_train, yr_train)
rf_pred = rf.predict(X_test)

rf_mae  = mean_absolute_error(yr_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(yr_test, rf_pred))
rf_r2   = r2_score(yr_test, rf_pred)

tscv = TimeSeriesSplit(n_splits=5)
rf_cv_r2 = cross_val_score(rf, X, y_reg, cv=tscv, scoring='r2').mean()

print(f"\n── Random Forest Regressor ──")
print(f"MAE:      {rf_mae:.4f}%")
print(f"RMSE:     {rf_rmse:.4f}%")
print(f"R²:       {rf_r2:.4f}")
print(f"CV R² (TimeSeriesSplit×5): {rf_cv_r2:.4f}")

# ─────────────────────────────────────────────
# 3. LOGISTIC REGRESSION CLASSIFIER
# ─────────────────────────────────────────────
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xte_sc = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
lr.fit(Xtr_sc, yc_train)
lr_pred = lr.predict(Xte_sc)
lr_prob = lr.predict_proba(Xte_sc)[:,1]

lr_acc = accuracy_score(yc_test, lr_pred)
lr_cv  = cross_val_score(lr, scaler.transform(X), y_clf, cv=tscv, scoring='accuracy').mean()

print(f"\n── Logistic Regression (High UR Classifier) ──")
print(f"Accuracy: {lr_acc:.4f}")
print(f"CV Accuracy (TimeSeriesSplit×5): {lr_cv:.4f}")
print(classification_report(yc_test, lr_pred, target_names=['Normal UR','High UR (Stress)']))

# ─────────────────────────────────────────────
# 4. ITERATIVE 12-MONTH FORECAST (2025)
# ─────────────────────────────────────────────
def forecast_sector(sector_name, steps=12):
    sec_df = df[df['Sector']==sector_name].sort_values(['Year','Month']).copy()
    history = sec_df.copy()
    enc = le.transform([sector_name])[0]
    preds = []

    for step in range(steps):
        last = history.tail(12)
        last_row = history.iloc[-1]
        mo = int(last_row['Month'] % 12) + 1
        yr = int(last_row['Year']) + (1 if mo == 1 else 0)
        t  = int(last_row['t']) + 1

        def lag(col, n):
            arr = history[col].values
            return arr[-n] if len(arr) >= n else np.nan

        def roll_mean(col, n):
            arr = history[col].values[-n:]
            return arr.mean() if len(arr) >= n else np.nan

        def roll_std(col, n):
            arr = history[col].values[-n:]
            return arr.std() if len(arr) >= 2 else 0.0

        row_feat = {
            'Sector_enc': enc, 'Month': mo, 't': t,
            'month_sin': np.sin(2*np.pi*mo/12),
            'month_cos': np.cos(2*np.pi*mo/12),
            'GDP_Growth_Rate':      lag('GDP_Growth_Rate', 1),
            'GDP_Growth_Rate_lag1': lag('GDP_Growth_Rate', 2),
            'LFPR_lag1': lag('LFPR',1), 'LFPR_lag2': lag('LFPR',2),
            'LFPR_lag3': lag('LFPR',3), 'LFPR_lag6': lag('LFPR',6),
            'LFPR_lag12': lag('LFPR',12),
            'UR_lag1': lag('UR',1), 'UR_lag2': lag('UR',2),
            'UR_lag3': lag('UR',3), 'UR_lag6': lag('UR',6),
            'UR_lag12': lag('UR',12),
            'WPR_lag1': lag('WPR',1), 'WPR_lag2': lag('WPR',2), 'WPR_lag3': lag('WPR',3),
            'UR_roll3':  roll_mean('UR',3),  'UR_roll6':  roll_mean('UR',6),
            'UR_roll3_std': roll_std('UR',3),
            'LFPR_roll3': roll_mean('LFPR',3), 'WPR_roll3': roll_mean('WPR',3),
            'UR_diff1':   history['UR'].iloc[-1]  - history['UR'].iloc[-2]  if len(history)>1 else 0,
            'LFPR_diff1': history['LFPR'].iloc[-1]- history['LFPR'].iloc[-2] if len(history)>1 else 0,
        }
        X_row = pd.DataFrame([row_feat])[FEATURES]
        ur_pred = float(rf.predict(X_row)[0])
        ur_stress = int(lr.predict(scaler.transform(X_row))[0])
        ur_prob   = float(lr.predict_proba(scaler.transform(X_row))[0,1])

        new_row = last_row.copy()
        new_row['Year'] = yr; new_row['Month'] = mo; new_row['t'] = t
        new_row['UR'] = ur_pred
        new_row['LFPR'] = lag('LFPR',1)
        new_row['WPR']  = lag('WPR',1)
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        preds.append({'Sector': sector_name, 'Year': yr, 'Month': mo,
                      'UR_RF_pred': round(ur_pred,3),
                      'UR_Stress_LR': ur_stress,
                      'Stress_Prob': round(ur_prob,3)})
    return preds

all_preds = []
for s in le.classes_:
    all_preds.extend(forecast_sector(s, steps=12))

forecast_df = pd.DataFrame(all_preds)
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
forecast_df['Month_Name'] = forecast_df['Month'].map(month_names)
print("\n── 2025 Iterative UR Forecasts ──")
print(forecast_df[['Sector','Month_Name','UR_RF_pred','Stress_Prob']].to_string(index=False))

# ─────────────────────────────────────────────
# 5.  VISUALISATIONS
# ─────────────────────────────────────────────
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
colors = {'Agriculture':'#4CAF50','IT':'#2196F3','Manufacturing':'#FF9800',
          'Medical':'#E91E63','Hospitality':'#9C27B0'}

fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor('#0F1117')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.32)

def style_ax(ax, title):
    ax.set_facecolor('#1A1D27')
    ax.tick_params(colors='#AAAAAA')
    ax.xaxis.label.set_color('#AAAAAA')
    ax.yaxis.label.set_color('#AAAAAA')
    ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
    for spine in ax.spines.values(): spine.set_edgecolor('#333344')

# ── 1. RF Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(yr_test.values, rf_pred, alpha=0.55, s=22, c=df_clean.iloc[split:]['Sector_enc'],
            cmap='tab10', edgecolors='none')
mn, mx = yr_test.min(), yr_test.max()
ax1.plot([mn,mx],[mn,mx],'--', color='#FF5722', lw=1.8, label='Perfect fit')
ax1.legend(facecolor='#1A1D27', labelcolor='white', fontsize=9)
ax1.set_xlabel('Actual UR (%)'); ax1.set_ylabel('Predicted UR (%)')
style_ax(ax1, f'RF Regression: Actual vs Predicted  (R²={rf_r2:.3f}, MAE={rf_mae:.3f}%)')

# ── 2. LR Confusion Matrix
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(yc_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax2,
            xticklabels=['Normal','High Stress'], yticklabels=['Normal','High Stress'],
            linewidths=0.5, linecolor='#0F1117', annot_kws={'color':'white','fontsize':13})
ax2.tick_params(colors='#AAAAAA')
style_ax(ax2, f'Logistic Regression Confusion Matrix  (Acc={lr_acc:.3f})')

# ── 3. Feature Importance
ax3 = fig.add_subplot(gs[1, :])
top_feats = pd.Series(rf.feature_importances_, index=FEATURES).nlargest(14)
palette = ['#7C4DFF' if 'UR' in f else '#00BCD4' if 'LFPR' in f else '#FF9800' for f in top_feats.index[::-1]]
bars = ax3.barh(top_feats.index[::-1], top_feats.values[::-1], color=palette, edgecolor='#0F1117')
for bar, val in zip(bars, top_feats.values[::-1]):
    ax3.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', color='#CCCCCC', fontsize=8.5)
ax3.set_xlabel('Importance Score')
# Legend patches
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor='#7C4DFF',label='UR features'),
                Patch(facecolor='#00BCD4',label='LFPR features'),
                Patch(facecolor='#FF9800',label='Other features')]
ax3.legend(handles=legend_elems, facecolor='#1A1D27', labelcolor='white', fontsize=9, loc='lower right')
style_ax(ax3, 'Random Forest – Top 14 Feature Importances')

# ── 4. Historical UR + 2025 Forecast
ax4 = fig.add_subplot(gs[2, :])
for sector in le.classes_:
    hist = df[df['Sector']==sector].copy()
    hist['date_f'] = hist['Year'] + (hist['Month']-0.5)/12
    ax4.plot(hist['date_f'], hist['UR'], color=colors[sector], lw=1.4, alpha=0.85)

    fut = forecast_df[forecast_df['Sector']==sector].copy()
    fut['date_f'] = fut['Year'] + (fut['Month']-0.5)/12
    ax4.plot(fut['date_f'], fut['UR_RF_pred'], color=colors[sector], lw=2.2,
             linestyle='--', marker='o', markersize=4, label=sector)
    ax4.fill_between(fut['date_f'],
                     fut['UR_RF_pred']-0.4, fut['UR_RF_pred']+0.4,
                     alpha=0.12, color=colors[sector])

ax4.axvline(2025, color='#FF5722', lw=1.5, linestyle=':', alpha=0.9)
ax4.text(2025.05, ax4.get_ylim()[0]+0.3, '2025 Forecast →', color='#FF5722', fontsize=9)
ax4.legend(loc='upper left', facecolor='#1A1D27', edgecolor='#333344',
           labelcolor='white', fontsize=9, ncol=2)
ax4.set_xlabel('Year'); ax4.set_ylabel('Unemployment Rate (%)')
style_ax(ax4, 'UR: Historical (solid) + 2025 Iterative RF Forecast (dashed) by Sector')

# ── 5. Model metrics bar
ax5 = fig.add_subplot(gs[3, 0])
metrics = ['RF\nR²', 'RF\nMAE%', 'LR\nAccuracy', 'LR\nCV Acc']
values  = [rf_r2, 1-rf_mae/10, lr_acc, lr_cv]   # normalise MAE for display
labels  = [f'{rf_r2:.3f}', f'{rf_mae:.3f}%', f'{lr_acc:.3f}', f'{lr_cv:.3f}']
bar_c   = ['#4CAF50','#2196F3','#FF9800','#E91E63']
brs = ax5.bar(metrics, values, color=bar_c, edgecolor='#0F1117', width=0.5)
for b, l in zip(brs, labels):
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             l, ha='center', color='white', fontsize=11, fontweight='bold')
ax5.set_ylim(0, 1.15); ax5.set_ylabel('Score')
style_ax(ax5, 'Model Performance Summary')

# ── 6. 2025 Forecast heatmap (stress probability)
ax6 = fig.add_subplot(gs[3, 1])
pivot = forecast_df.pivot(index='Sector', columns='Month_Name', values='Stress_Prob')
mo_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot[[m for m in mo_order if m in pivot.columns]]
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax6,
            linewidths=0.4, linecolor='#0F1117',
            vmin=0, vmax=1, annot_kws={'size':8,'color':'white'})
ax6.tick_params(colors='#AAAAAA'); ax6.set_xlabel(''); ax6.set_ylabel('')
style_ax(ax6, '2025 Employment Stress Probability (LR) by Sector & Month')

fig.suptitle('India Employment Prediction  ·  Random Forest + Logistic Regression',
             color='white', fontsize=15, fontweight='bold', y=0.995)

plt.savefig('employment_model_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0F1117')
plt.close()
print("\nPlot saved.")

forecast_df.to_csv('employment_2025_predictions.csv', index=False)
print("Forecast CSV saved.")