import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px

# 1. 加载数据
file_path = r'E:\Experiments\IntrusionDetection\NSL_KDD\KDDTrain+.txt'
data = pd.read_csv(file_path, header=None)

# 2. 数据预处理
X = data.iloc[:, :-2]  # 所有特征
y = data.iloc[:, -2:-1]  # 标签（如果需要）

# 假设第2、3、4列是分类特征（索引为1, 2, 3）
categorical_features = [1, 2, 3]
numerical_features = [col for col in X.columns if col not in categorical_features]

# 使用ColumnTransformer对分类特征进行独热编码，数值特征保持不变
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(X)

# 3. 特征归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# 4. K-means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_  # 聚类标签

# 5. 获取每一类的数量
cluster_counts = np.bincount(labels)
for cluster_id, count in enumerate(cluster_counts):
    print(f"Cluster {cluster_id}: {count} samples")

# 6. 可视化（3D PCA）
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

fig = px.scatter_3d(
    X_pca, x=0, y=1, z=2, color=labels,
    title='3D PCA of K-means Clustering',
    labels={'0': 'PCA 1', '1': 'PCA 2', '2': 'PCA 3'},
    opacity=0.7
)
fig.show()

# 7. 2D PCA可视化（更清晰）
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

fig_2d = px.scatter(
    X_pca_2d, x=0, y=1, color=labels,
    title='2D PCA of K-means Clustering',
    labels={'0': 'PCA 1', '1': 'PCA 2'},
    opacity=0.7
)
fig_2d.show()