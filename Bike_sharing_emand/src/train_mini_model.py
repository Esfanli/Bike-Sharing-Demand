# %% [markdown]
# <img src="https://webna.ir/wp-content/uploads/2018/08/%D9%85%DA%A9%D8%AA%D8%A8-%D8%AE%D9%88%D9%86%D9%87.png" width=50% />

# %% [markdown]
# <div class="alert alert-block alert-success">
#     <h1 align="center">Machine Learning in Python</h1>
#     <h3 align="center">Bike Sharing</h3>
# </div>

# %% [markdown]
# <img src = "https://storage.googleapis.com/kaggle-competitions/kaggle/3948/media/bikes.png" width=80%>

# %% [markdown]
# ## Importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV


# %% [markdown]
# ## Load and Prepare Data

# %%
df = pd.read_csv('../data/train.csv')
df.head(10)

# %% [markdown]
# ## EDA

# %%
print("Shape: ",df.shape)
print()
df.info()

# %%
df.describe().T

# %%
print("Missing value:")
print(df.isnull().sum())
print("Duplicates rows: ", df.duplicated().sum())

# %%
(df['casual'] + df['registered'] == df['count'].all())

# %%
for col in ['season' , 'holiday' , 'workingday' , 'weather']:
    print(col , "->" , sorted(df[col].unique()))

# %%
# بررسی تعداد رکوردهای هر دسته‌ی آب‌وهوا و فصل

print("Weather counts:")
print(df['weather'].value_counts().sort_index())
print()
print("Season counts:")
print(df['season'].value_counts().sort_index())

# %% [markdown]
# ## Data Preprocessing
# 

# %%
# datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df = df.drop(columns=['datetime'])


# %%
df['weather'] = df['weather'].replace({4: 3})

# %%
df = df.drop(columns=['casual','registered'])

# %%
df = pd.get_dummies(df , columns=['season' , 'weather'] , drop_first=False)

# %%
bool_cols = df.select_dtypes(include='bool').columns


df[bool_cols] = df[bool_cols].astype(int)

#df.head()

# %%
# Split
x = df.drop(columns=['count'])
y = df['count']
print(x.shape)
print(y.shape)
#print("\n",x.columns.tolist())


# %%
x_train , x_val , y_train , y_val = train_test_split(
    x,y,
    test_size=0.2 ,
    random_state=42
)
print("X_t :",x_train.shape , "Y_t" , y_train.shape)
print("X_v :",x_val.shape , "Y_v" , y_val.shape)

# %%
#scaling
scaler = StandardScaler()
scaler.fit(x_train)
print(len(scaler.mean_))

# %%
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

print(x_train_scaled.shape)
print(x_val_scaled.shape)

# %%
print(np.round(x_train_scaled.mean(axis=0), 3))
print()
print(np.round(x_train_scaled.std(axis=0), 3))

# %% [markdown]
# ## _______________________________________________________________________________________________________________________________________________

# %% [markdown]
# ## Strorytelling - Visualization

# %%
# میانگین تعداد اجاره‌ها در هر ساعت از شبانه‌روز
plt.figure(figsize=(12, 5))
sns.barplot(x='hour', y='count', data=df)
plt.title("Average Bike Rentals by Hour of Day")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Average Count")
plt.show()

# %%
# مقایسه‌ی الگوی ساعتی در روزهای کاری و غیرکاری
plt.figure(figsize=(14, 6))
sns.pointplot(x='hour', y='count', hue='workingday', data=df,
              palette={0: '#e74c3c', 1: '#3498db'})
plt.title("Average Bike Rentals by Hour: Working Day vs Weekend/Holiday")
plt.xlabel("Hour of Day (0-23)")
plt.ylabel("Average Count")
plt.legend(title='Working Day', labels=['No (Weekend/Holiday)', 'Yes (Working Day)'])
plt.grid(axis='y', alpha=0.3)
plt.show()

# %%
# --- فصل و آب‌وهوا ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# نمودار فصل
sns.boxplot(x='season', y='count', data=df, ax=axes[0])
axes[0].set_title("Bike Rentals by Season")
axes[0].set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
axes[0].set_xlabel("Season")
axes[0].set_ylabel("Count")

# نمودار آب‌وهوا
sns.boxplot(x='weather', y='count', data=df, ax=axes[1])
axes[1].set_title("Bike Rentals by Weather")
axes[1].set_xticklabels(['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'])
axes[1].set_xlabel("Weather")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# %%
# --- فیچرهای عددی vs target ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# temp vs count
axes[0].scatter(df['temp'], df['count'], alpha=0.2, s=8, color='#e74c3c')
axes[0].set_title("Temperature vs Rentals")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Count")

# humidity vs count
axes[1].scatter(df['humidity'], df['count'], alpha=0.2, s=8, color='#3498db')
axes[1].set_title("Humidity vs Rentals")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("Count")

# windspeed vs count
axes[2].scatter(df['windspeed'], df['count'], alpha=0.2, s=8, color='#27ae60')
axes[2].set_title("Windspeed vs Rentals")
axes[2].set_xlabel("Windspeed")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.show()

# %%
# --- روند تعداد اجاره‌ها در طول زمان ---
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['count'], linewidth=0.3, color='#2c3e50')
plt.title("Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Count")
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## Train your model (Regression/Regressor)

# %%
LR = LinearRegression()
LR.fit(x_train_scaled , y_train)
y_pred_LR = LR.predict(x_val_scaled)

# %%
KNN = KNeighborsRegressor(n_neighbors=5,algorithm='brute',n_jobs=-1)
KNN.fit(x_train_scaled,y_train)
y_pred_KNN = KNN.predict(x_val_scaled)

# %%
# KNN GSC
# از این کتابخونه جی اس سی استفادهکردم که میشه به یک باره باهاش چندین پارامتر مختلف چک میشه و بهترین رو انتخاب میکنه دیگخ نیاز نیست دستی چندین بار تست کنیم
from sklearn.model_selection import GridSearchCV

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10, 15, 20, 30],
    'weights': ['uniform', 'distance']
}

grid_knn = GridSearchCV(
    KNeighborsRegressor(algorithm='brute', n_jobs=-1),
    param_grid_knn,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_knn.fit(x_train_scaled, y_train)

print("Best KNN params:", grid_knn.best_params_)
print("Best KNN CV R²:", grid_knn.best_score_)
# خب زیاد بهبود پیدانکرد و بهتره  همینجور بزاریمش کنار بقیه مدلا
# دلایش میتونه دیتای نویزی یا ابعاد بالا باشه چون 18 فیچر میشه 18 بعد و در 18 بعد با بعضی فیچرای نویزی و الگوری پیچیده بعضی فیچرا باشه
# فعلا همون چنگل تصادفی بهتریم عملکردو داشته
# ولی بریم سراغ نهایی کردنش و ببینم بهتر از سری قبل شده یا نه پون ما تا اینحا آردو رو دیدیم فقط

# %%
# KNN tuning
KNN_tuned = KNeighborsRegressor(
    n_neighbors=7,
    weights='distance',
    algorithm='brute',
    n_jobs=-1
)
KNN_tuned.fit(x_train_scaled, y_train)
y_pred_KNN_tuned = KNN_tuned.predict(x_val_scaled)

# %%
DT = DecisionTreeRegressor(random_state=42)
DT.fit(x_train , y_train)
y_pred_DT = DT.predict(x_val)

# %%
# DT tuninig
depths = [2, 3, 5, 7, 10, 15, 20]
results_dt = []

for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(x_train, y_train)
    
    r2_train = r2_score(y_train, dt.predict(x_train))
    r2_val   = r2_score(y_val, dt.predict(x_val))
    
    results_dt.append({
        'max_depth': d,
        'Train R²': round(r2_train, 4),
        'Val R²':   round(r2_val, 4),
        'Gap':      round(r2_train - r2_val, 4)
    })

df_dt = pd.DataFrame(results_dt)
print(df_dt.to_string(index=False))
# max depth = 10 is best

# %%
#  DT : max_depth=10 
DT_tuned = DecisionTreeRegressor(max_depth=10, random_state=42)
DT_tuned.fit(x_train, y_train)
y_pred_DT_tuned = DT_tuned.predict(x_val)



# %%
RF = RandomForestRegressor(
    n_estimators = 100,
    random_state = 42,
    n_jobs=-1
)
RF.fit(x_train , y_train)
y_pred_RF = RF.predict(x_val)

# %%
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_rf.fit(x_train, y_train)

print("Best RF params:", grid_rf.best_params_)
print("Best RF CV R²:", grid_rf.best_score_)

# %%
# RF tuning
RF_tuned = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
RF_tuned.fit(x_train, y_train)
y_pred_RF_tuned = RF_tuned.predict(x_val)


# %% [markdown]
# ## __________________________________________________________________________________________________

# %% [markdown]
# ## Test Result

# %%
MAE = mean_absolute_error(y_val,y_pred_LR)
RMSE = np.sqrt(mean_squared_error(y_val,y_pred_LR))
R2 = r2_score(y_val , y_pred_LR)
print('MAE: ' , MAE)
print('RMSE: ' , RMSE)
print('R2: ' , R2)
# بهبودش نمیدم چون بفیه الگوریتما عملکرد بهتری دارن پس نیاز نیست وقتمو بزارم برای بهبود عملکرد رگرسیون ساده

# %%
MAE_KNN = mean_absolute_error(y_val , y_pred_KNN)
RMSE_KNN = np.sqrt(mean_squared_error(y_val , y_pred_KNN))
R2_KNN = r2_score(y_val , y_pred_KNN)
print("KNN regressor (k=5)")
print(f"MAE_KNN: {MAE_KNN:.2f}")
print(f"RMSE_KNN: {RMSE_KNN:.2f}")
print(f"R2_KNN: {R2_KNN:.2f}")

# %%
MAE_KNN_tuned  = mean_absolute_error(y_val, y_pred_KNN_tuned)
RMSE_KNN_tuned = np.sqrt(mean_squared_error(y_val, y_pred_KNN_tuned))
R2_KNN_tuned   = r2_score(y_val, y_pred_KNN_tuned)

print("KNN Tuned (k=7, distance):")
print(f"  MAE : {MAE_KNN_tuned:.2f}")
print(f"  RMSE: {RMSE_KNN_tuned:.2f}")
print(f"  R²  : {R2_KNN_tuned:.4f}")
# و بعله از مدت قبلی بهتر شد و ارز داره اینو به جای حالت پایه استفاده کنیم چون عملکرد بهتری داره
# دلیل تفاوت آردو در مرحله قبل و اینجا اینه که اوتجا مبانگین 5 فولد مختلف گزارش داده میشد 
#فولد ولیدیشن ما که تصادفی از ترین جدا شده و شانسی راحتر از میانگین بوده
#پس سی وی واقعبینانه تره و ولیدیشن ما شانسی تر بوده
# منظوز از شانسی اینه که اسپلیتی که با رندوم استیت 42 ساختیم راحتر از میانگین بوده
# شانسی نبوده در واقع جی سی وی واقها بهترین پارامترو پیداکرده و کی ان ان بهتر شده
# موقع پرازش باید آردو0.5653 گزارش داه یشه چون واقع بینانست میانگین پنج حالت مختلف نه یه شانس خوب
# در واقع جون ولیدیشن تصادفی انتخاب شده یعنی شانسمون خوب بوده


# %%
MAE_DT = mean_absolute_error(y_val,y_pred_DT)
RMSE_DT = np.sqrt(mean_squared_error(y_val , y_pred_DT))
R2_DT = r2_score(y_val , y_pred_DT)
print(f"MAE_DT: {MAE_DT:.2f}")
print(f"RMSE_DT: {RMSE_DT:.2f}")
print(f"R2_DT: {R2_DT:.2f}")

# %%
MAE_DT_tuned  = mean_absolute_error(y_val, y_pred_DT_tuned)
RMSE_DT_tuned = np.sqrt(mean_squared_error(y_val, y_pred_DT_tuned))
R2_DT_tuned   = r2_score(y_val, y_pred_DT_tuned)

print("Decision Tree (max_depth=10):")
print(f"  MAE : {MAE_DT_tuned:.2f}")
print(f"  RMSE: {RMSE_DT_tuned:.2f}")
print(f"  R²  : {R2_DT_tuned:.4f}")
# خب عمق 10 مدل رو ضعیفتر کرد پس تیونینگ برای دیسیژن تری فقط بدترش کرد
# ارزش جنگل تصادفی بالاتره چون در جالت پایش از جالت پایه و بهبود یافته این بهتر بوده
# نتیجه : تیون همیشه بهترین کار نیست

# %%
MAE_RF = mean_absolute_error(y_val,y_pred_RF)
RMSE_RF = np.sqrt(mean_squared_error(y_val , y_pred_RF))
R2_RF = r2_score(y_val , y_pred_RF)
print(f"MAE_RF: {MAE_RF:.2f}")
print(f"RMSE_RF: {RMSE_RF:.2f}")
print(f"R2_RF: {R2_RF:.2f}")

# %%
MAE_RF_tuned  = mean_absolute_error(y_val, y_pred_RF_tuned)
RMSE_RF_tuned = np.sqrt(mean_squared_error(y_val, y_pred_RF_tuned))
R2_RF_tuned   = r2_score(y_val, y_pred_RF_tuned)

print("Random Forest Tuned:")
print(f"  MAE : {MAE_RF_tuned:.2f}")
print(f"  RMSE: {RMSE_RF_tuned:.2f}")
print(f"  R²  : {R2_RF_tuned:.4f}")
# خب بهبودی نداشتیم پون رندوم فارست خودش ذاتا مدل قوییه و پارامترای پیش فرض ار قبل خوب بودن
# پس گرید سرچ عملا بی فایده بوده و 36 ترکیب مختلف هیچکدوم بهتر از پیش فرض نبود
# درسته آردو یه چیز دیگه شد ولی با آردویی که داشتیم تقریبا یکی بود تفاوتشون ناچیزه
# و میتونه دلیلش اینم باشه که به سقف دیتا رسیدیم و بقیه خطاها نویز داتی دیتاها و فیچرا هستن
# پس همیشه تیونینگ بهتر نیست ولی انحام میشه که شاید بهتر باشه و تازه سی وی برای گزارشم بهمون میده
# ما همینو نگه میداریم چون یکمم بهتر شده

# %%
# check all models

# --- Train predictions ---
y_train_pred_LR  = LR.predict(x_train_scaled)
y_train_pred_KNN = KNN.predict(x_train_scaled)
y_train_pred_DT  = DT.predict(x_train)
y_train_pred_RF  = RF.predict(x_train)

# --- Train R² ---
R2_train_LR  = r2_score(y_train, y_train_pred_LR)
R2_train_KNN = r2_score(y_train, y_train_pred_KNN)
R2_train_DT  = r2_score(y_train, y_train_pred_DT)
R2_train_RF  = r2_score(y_train, y_train_pred_RF)

# --- جدول Overfitting ---
overfit_check = pd.DataFrame({
    'Model':      ['Linear', 'KNN', 'Decision Tree', 'Random Forest'],
    'Train R²':   [R2_train_LR, R2_train_KNN, R2_train_DT, R2_train_RF],
    'Val R²':     [R2, R2_KNN, R2_DT, R2_RF],
    'Gap (Train-Val)': [R2_train_LR - R2, R2_train_KNN - R2_KNN, R2_train_DT - R2_DT, R2_train_RF - R2_RF]
})

print(overfit_check.to_string(index=False))

# %%
# ==============================
# Cross Validation
# ==============================
from sklearn.model_selection import cross_val_score

cv_scores_LR  = cross_val_score(LR,        x_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
cv_scores_KNN = cross_val_score(KNN_tuned, x_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
cv_scores_DT  = cross_val_score(DT_tuned,  x_train,        y_train, cv=5, scoring='r2', n_jobs=-1)
cv_scores_RF  = cross_val_score(RF_tuned,  x_train,        y_train, cv=5, scoring='r2', n_jobs=-1)

print("=" * 60)
print("Cross-Validation R² (5-fold, on X_train only)")
print("=" * 60)
print(f"  Linear Regression : {cv_scores_LR.mean():.4f}  ±  {cv_scores_LR.std():.4f}")
print(f"  KNN (tuned)       : {cv_scores_KNN.mean():.4f}  ±  {cv_scores_KNN.std():.4f}")
print(f"  Decision Tree     : {cv_scores_DT.mean():.4f}  ±  {cv_scores_DT.std():.4f}")
print(f"  Random Forest     : {cv_scores_RF.mean():.4f}  ±  {cv_scores_RF.std():.4f}")

# %%
# --- Feature Importance -> Random Forest ---
importances = RF_tuned.feature_importances_

feature_imp = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("Top 18 Most Important Features:")
print(feature_imp.head(18).to_string(index=False))

# --- نمودار ---
plt.figure(figsize=(15, 9))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_imp.head(18),
    palette='viridis'
)
plt.title("Top 18 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## data preprossing for test set

# %%
test_df = pd.read_csv("../data/test.csv")
test_datetime = test_df['datetime'].copy()

test_df['datetime'] = pd.to_datetime(test_df['datetime'])
test_df['year']      = test_df['datetime'].dt.year
test_df['month']     = test_df['datetime'].dt.month
test_df['day']       = test_df['datetime'].dt.day
test_df['hour']      = test_df['datetime'].dt.hour
test_df['dayofweek'] = test_df['datetime'].dt.dayofweek

test_df['weather'] = test_df['weather'].replace({4: 3})
test_df = test_df.drop(columns=['datetime'])

test_df = pd.get_dummies(test_df, columns=['season', 'weather'], drop_first=False)
bool_cols_test = test_df.select_dtypes(include='bool').columns
test_df[bool_cols_test] = test_df[bool_cols_test].astype(int)

test_df = test_df[x_train.columns]

print("Test shape:", test_df.shape)
print("Columns match:", list(test_df.columns) == list(x_train.columns))

# %% [markdown]
# ## final prediction on test set

# %%
y_test_pred = RF_tuned.predict(test_df)

print("Predictions shape:", y_test_pred.shape)
print("First 10 predictions:", y_test_pred[:10])
print()
print("Stats:")
print(f"  Min : {y_test_pred.min():.2f}")
print(f"  Max : {y_test_pred.max():.2f}")
print(f"  Mean: {y_test_pred.mean():.2f}")

# %% [markdown]
# ## _______________________________________________________________________________________________

# %% [markdown]
# ## The End

# %%
submission = pd.DataFrame({
    'datetime': test_datetime,
    'count':    y_test_pred.round().astype(int)
})

submission.to_csv("submission_01.csv", index=False)
print("✅ submission_01.csv saved.")
print(submission.head())

# %%
# end check
sample = pd.read_csv("../data/sampleSubmission.csv")

print("Sample shape:", sample.shape)
print("Submission shape:", submission.shape)
print("Same length?", len(sample) == len(submission))
print("Same datetime order?", (sample['datetime'] == submission['datetime']).all())

# %% [markdown]
# ## --------------------------------------------------------
# 

# %%
# --- پیش‌بینی روی کل train ---
y_train_pred_full = RF_tuned.predict(x_train)

# --- مقایسه ۲۰ تا رکورد اول ---
comparison = pd.DataFrame({
    'Actual':    y_train.values[:20],
    'Predicted': y_train_pred_full[:20].round().astype(int),
    'Error':     (y_train.values[:20] - y_train_pred_full[:20]).round().astype(int)
})

print(comparison)
print(f"\nMean Absolute Error: {mean_absolute_error(y_train, y_train_pred_full):.2f}")

# %%
def predict_rentals(season, holiday, workingday, weather, temp, atemp, humidity, windspeed,
                    year, month, day, hour, dayofweek):
    """
    ورودی: ویژگی‌های آب‌وهوا و زمان
    خروجی: تعداد اجاره‌ی پیش‌بینی‌شده
    """
    df_input = pd.DataFrame({
        'season': [season], 'holiday': [holiday], 'workingday': [workingday],
        'weather': [weather], 'temp': [temp], 'atemp': [atemp],
        'humidity': [humidity], 'windspeed': [windspeed],
        'year': [year], 'month': [month], 'day': [day],
        'hour': [hour], 'dayofweek': [dayofweek]
    })
    
    df_input = pd.get_dummies(df_input, columns=['season', 'weather'], drop_first=False)
    bool_cols_ = df_input.select_dtypes(include='bool').columns
    df_input[bool_cols_] = df_input[bool_cols_].astype(int)
    df_input = df_input.reindex(columns=x_train.columns, fill_value=0)
    
    return RF_tuned.predict(df_input)[0]


# --- تست چند سناریو ---
print("سناریو ۱ - ساعت ۸ صبح، روز کاری، بهار، صاف:")
print(f"  → {predict_rentals(1, 0, 1, 1, 20, 22, 60, 10, 2012, 4, 15, 8, 0):.0f} اجاره")
print()

print("سناریو ۲ - ساعت ۸ صبح، آخر هفته، بهار، صاف:")
print(f"  → {predict_rentals(1, 0, 0, 1, 20, 22, 60, 10, 2012, 4, 15, 8, 6):.0f} اجاره")
print()

print("سناریو ۳ - ساعت ۱۷ عصر، روز کاری، تابستان، صاف:")
print(f"  → {predict_rentals(2, 0, 1, 1, 30, 33, 50, 10, 2012, 7, 15, 17, 0):.0f} اجاره")
print()

print("سناریو ۴ - ساعت ۳ صبح، روز کاری، زمستان، بارانی:")
print(f"  → {predict_rentals(4, 0, 1, 3, 5, 3, 90, 15, 2012, 1, 15, 3, 0):.0f} اجاره")

# %%
print("سناریو ۱ - ۲۰۱۱:")
print(f"  → {predict_rentals(1, 0, 1, 1, 20, 22, 60, 10, 2011, 4, 15, 8, 0):.0f} اجاره")
print()

print("سناریو ۳ - ۲۰۱۱:")
print(f"  → {predict_rentals(2, 0, 1, 1, 30, 33, 50, 10, 2011, 7, 15, 17, 0):.0f} اجاره")


