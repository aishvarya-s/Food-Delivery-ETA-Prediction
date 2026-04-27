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
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()
def graph2(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    df = df.copy()

    # 🔹 Encode traffic levels
    traffic_map = {'Low':1, 'Medium':2, 'High':3, 'Jam':4}
    df['traffic'] = df['Road_traffic_density'].map(traffic_map)

    # remove missing
    df = df.dropna(subset=['distance_euclidean', 'traffic'])

    # 🔹 Prepare data
    X = df[['distance_euclidean', 'traffic']]
    X_scaled = StandardScaler().fit_transform(X)

    # 🔹 DBSCAN
    labels = DBSCAN(eps=0.4, min_samples=8).fit_predict(X_scaled)

    clusters = labels != -1
    noise = labels == -1

    # 🔹 Add jitter (for visibility)
    jitter = df['traffic'] + np.random.uniform(-0.1, 0.1, len(df))

    # ✅ plot clusters
    ax.scatter(
        df['distance_euclidean'][clusters],
        jitter[clusters],
        c=labels[clusters],
        cmap='viridis',
        s=15
    )

    # ✅ plot noise
    ax.scatter(
        df['distance_euclidean'][noise],
        jitter[noise],
        c='red',
        s=20,
        label='Noise'
    )

    # 🔹 axis labels
    ax.set_yticks([1,2,3,4])
    ax.set_yticklabels(['Low','Medium','High','Jam'])

    ax.set_title("Distance vs Traffic")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Traffic")

    # 🔥 important fix (same as graph1)
    ax.set_xlim(0, df['distance_euclidean'].quantile(0.99))

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

    ax.set_title("Deliveries vs Time")
    ax.set_xlabel("Number of Deliveries")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()
def graph6(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    df = df.copy()

    # encode weather
    weather_map = {
        'Sunny': 1,
        'Cloudy': 2,
        'Fog': 3,
        'Stormy': 4,
        'Sandstorms': 5,
        'Windy': 6
    }

    df['weather'] = df['Weather_conditions'].map(weather_map)

    df = df.dropna(subset=['weather', 'Time_taken (min)'])

    # prepare data
    X = df[['weather', 'Time_taken (min)']]
    X_scaled = StandardScaler().fit_transform(X)

    labels = DBSCAN(eps=0.4, min_samples=8).fit_predict(X_scaled)

    clusters = labels != -1
    noise = labels == -1

    # jitter
    jitter = df['weather'] + np.random.uniform(-0.1, 0.1, len(df))

    # clusters
    ax.scatter(
        jitter[clusters],
        df['Time_taken (min)'][clusters],
        c=labels[clusters],
        cmap='viridis',
        s=15
    )

    # noise
    ax.scatter(
        jitter[noise],
        df['Time_taken (min)'][noise],
        c='red',
        s=20,
        label='Noise'
    )

    ax.set_yticks(range(1, 7))
    ax.set_yticklabels(['Sunny','Cloudy','Fog','Stormy','Sandstorms','Windy'])

    ax.set_title("Weather vs Time")
    ax.set_xlabel("Weather")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()
def graph4(df, ax):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    df = df.copy()

    # remove missing values
    df = df.dropna(subset=['distance_euclidean', 'multiple_deliveries'])

    # prepare data
    X = df[['distance_euclidean', 'multiple_deliveries']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # plot each cluster (NO jitter)
    for i in range(3):
        ax.scatter(
            df['distance_euclidean'][labels == i],
            df['multiple_deliveries'][labels == i],
            label=f'Cluster {i}',
            s=12,
            alpha=0.8
        )

    # plot centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c='red',
        s=80,
        marker='X',
        label='Centroids'
    )

    # labels
    ax.set_title("Distance vs Deliveries (KMeans)")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Number of Deliveries")

    # clean axis
    ax.set_xlim(0, df['distance_euclidean'].quantile(0.99))
    ax.set_yticks([0, 1, 2, 3])

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
def graph6(df, ax):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    df = df.copy()

    # 🔹 encode weather (same order as labels)
    weather_map = {
        'Sunny': 1,
        'Cloudy': 2,
        'Fog': 3,
        'Stormy': 4,
        'Sandstorms': 5,
        'Windy': 6
    }

    df['weather'] = df['Weather_conditions'].map(weather_map)

    # remove missing
    df = df.dropna(subset=['weather', 'Time_taken (min)'])

    # 🔹 prepare data
    X = df[['weather', 'Time_taken (min)']]
    X_scaled = StandardScaler().fit_transform(X)

    # 🔹 DBSCAN
    labels = DBSCAN(eps=0.4, min_samples=8).fit_predict(X_scaled)

    clusters = labels != -1
    noise = labels == -1

    # 🔹 jitter (for visibility)
    jitter = df['weather'] + np.random.uniform(-0.1, 0.1, len(df))

    # ✅ plot clusters
    ax.scatter(
        jitter[clusters],
        df['Time_taken (min)'][clusters],
        c=labels[clusters],
        cmap='viridis',
        s=15,
        alpha=0.7
    )

    # ✅ plot noise
    ax.scatter(
        jitter[noise],
        df['Time_taken (min)'][noise],
        c='red',
        s=20,
        label='Noise'
    )

    # 🔥 FIXED LABELING (clean + readable)
    ax.set_xticks([1,2,3,4,5,6])
    ax.set_xticklabels(
        ['Sunny','Cloudy','Fog','Stormy','Sand','Windy'],
        rotation=45,
        ha='right'
    )

    ax.set_title("Weather vs Time")
    ax.set_xlabel("Weather")
    ax.set_ylabel("Time Taken (min)")

    ax.legend()

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
    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.flatten()

    graph1(df_cluster, axes[0])
    graph2(df_cluster, axes[1])
    graph3(df_cluster, axes[2])
    graph4(df_cluster, axes[3])
    graph5(df_cluster, axes[4])
    graph6(df_cluster, axes[5])
    plt.tight_layout()
    plt.show()
    plt.close('all')