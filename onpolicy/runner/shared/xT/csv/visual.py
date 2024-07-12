import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 경로
csv_file = '/home/uosai/Desktop/on-policy/onpolicy/runner/shared/xT/xTModel_xT.csv'

# CSV 파일 읽기
df = pd.read_csv(csv_file, header=None, skiprows=1)

log_df = np.log1p(df)

# 히트맵 그리기
plt.figure(figsize=(10, 8))
sns.heatmap(log_df, cmap="viridis", cbar=True)

# 제목과 축 레이블 추가
plt.title('Heatmap of CSV Data')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 히트맵 표시
plt.show()


