import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans


# -------------------------------
# GRAPH 1: DISTANCE vs TIME
# -------------------------------
def graph1(df, ax):
    X = df[['distance_euclidean', 'Time_taken (min)']]
    X_scaled = StandardScaler().fit_transform(X)

    labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(X_scaled)

    clusters = labels != -1
    noise = labels == -1

    # ✅ plot clusters
    scatter = ax.scatter(
        df['distance_euclidean'][clusters],
        df['Time_taken (min)'][clusters],
        c=labels[clusters],
        cmap='viridis',
        s=20
    )

    # ✅ plot outliers (RED)
    ax.scatter(
        df['distance_euclidean'][noise],
        df['Time_taken (min)'][noise],
        c='red',
        s=20,
        label='Noise'
    )

    ax.set_xlim(0, df['distance_euclidean'].quantile(0.99))

    ax.set_title("Distance vs Time")
    ax.set_xlabel(" Distance (m)")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()

def graph3(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    df = df.copy()

    # remove missing
    df = df.dropna(subset=['multiple_deliveries', 'Time_taken (min)'])

    # 🔹 Prepare data
    X = df[['multiple_deliveries', 'Time_taken (min)']]
    X_scaled = StandardScaler().fit_transform(X)

    # 🔹 DBSCAN
    labels = DBSCAN(eps=0.4, min_samples=8).fit_predict(X_scaled)

    clusters = labels != -1
    noise = labels == -1

    # 🔹 jitter (so points don’t overlap vertically)
    jitter = df['multiple_deliveries'] + np.random.uniform(-0.1, 0.1, len(df))

    # ✅ plot clusters
    ax.scatter(
        jitter[clusters],
        df['Time_taken (min)'][clusters],
        c=labels[clusters],
        cmap='viridis',
        s=15
    )

    # ✅ plot noise
    ax.scatter(
        jitter[noise],
        df['Time_taken (min)'][noise],
        c='red',
        s=20,
        label='Noise'
    )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0', '1', '2'])
    ax.set_title("Deliveries vs Time")
    ax.set_xlabel("Number of Deliveries")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()

def graph4(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    df = df.copy()

    # remove missing
    df = df.dropna(subset=['distance', 'multiple_deliveries'])

    # 🔹 features
    X = df[['distance', 'multiple_deliveries']]

    # 🔹 scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 KMeans (2 clusters like your original)
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # 🔹 jitter (IMPORTANT)
    delivery_jitter = df['multiple_deliveries'] + np.random.uniform(-0.1, 0.1, len(df))

    # 🔹 scatter (single call, not loop)
    scatter = ax.scatter(
        df['distance'],
        delivery_jitter,
        c=df['cluster'],
        cmap='viridis',
        s=30,
        alpha=0.7
    )

    # 🔹 centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        s=100,
        marker='X',
        label='Centroids'
    )

    # 🔹 labels
    ax.set_title("K-Means Clustering (Distance + Deliveries)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Number of Deliveries")

    # 🔹 clean y-axis
    ax.set_yticks([0, 1, 2, 3])

    # 🔹 colorbar
    ax.figure.colorbar(scatter, ax=ax, label='Cluster')

    ax.legend()
def graph5(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    df = df.copy()

    # encode festival
    df['festival'] = df['Festival'].map({'No': 0, 'Yes': 1})

    df = df.dropna(subset=['festival', 'Time_taken (min)'])

    # prepare data
    X = df[['festival', 'Time_taken (min)']]
    X_scaled = StandardScaler().fit_transform(X)

    # DBSCAN
    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)

    df['cluster'] = labels

    # 🔥 jitter (same as your notebook)
    jitter = df['festival'] + np.random.uniform(-0.3, 0.3, len(df))

    # ✅ plot ALL points (no manual noise split)
    scatter = ax.scatter(
        jitter,
        df['Time_taken (min)'],
        c=df['cluster'],
        cmap='viridis',
        s=30
    )

    ax.set_xticks([0,1])
    ax.set_xticklabels(['No Festival', 'Festival'])

    ax.set_title("Festival vs Time (DBSCAN)")
    ax.set_xlabel("Festival")
    ax.set_ylabel("Time Taken (min)")

    # ✅ colorbar (important)
    plt.colorbar(scatter, ax=ax, label='Cluster')

def main(df):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = df.copy()

    # remove invalid coords
    df = df[
        (df['Restaurant_latitude'] != 0) &
        (df['Delivery_location_latitude'] != 0)
    ]

    # Euclidean distance
    df['distance_euclidean'] = np.sqrt(
        (df['Restaurant_latitude'] - df['Delivery_location_latitude'])**2 +
        (df['Restaurant_longitude'] - df['Delivery_location_longitude'])**2
    )

    # ✅ scale it
    df['distance_euclidean'] *= 100

    # handle time column
    if 'Actual_Time (min)' in df.columns:
        df['Time_taken (min)'] = df['Actual_Time (min)']
    elif 'Time_taken_min' in df.columns:
        df['Time_taken (min)'] = df['Time_taken_min']

    df['Time_taken (min)'] = pd.to_numeric(df['Time_taken (min)'], errors='coerce')

    df = df.dropna(subset=['distance_euclidean', 'Time_taken (min)'])

    # ✅ filter using Euclidean
    df_cluster = df[df['distance_euclidean'] > 50].copy()

    print("Min euclidean distance:", df_cluster['distance_euclidean'].min())

    # plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    graph1(df_cluster, axes[0])
    graph3(df_cluster, axes[1])
    graph4(df_cluster, axes[2])
    graph5(df_cluster, axes[3])
    plt.tight_layout()
    plt.show()