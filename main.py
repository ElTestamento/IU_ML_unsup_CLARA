import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from nltk.corpus import stopwords
from wordcloud import WordCloud
import gc

# Definiere Stoppwörter am Anfang
CUSTOM_STOP_WORDS = list(stopwords.words('english')) + [
    'show', 'results', 'x', 'et', 'al', 'using', 'one', 'two', 'three',
    'model', 'theory', 'non', 'system', 'systems', 'n', 'k', 'p', 'b',
    'groups', 'production', 'theorem', 'study', 'method', 'based', 'analysis',
    'new', 'approach', 'paper', 'methods', 'data', 'models', 'random', 'algebras',
    'states', 'induced', 'gauge', 'manifolds', 'function', 'functions', 'order',
    'properties', 'type', 'equations', 'problem'
]

# Für die WordCloud als Set
STOP_WORDS_SET = set(CUSTOM_STOP_WORDS)

class CLARA:
    def __init__(self, n_clusters, n_samples=500, n_sampling_runs=5, random_state=None):
        self.n_clusters = n_clusters
        self.n_samples = n_samples
        self.n_sampling_runs = n_sampling_runs
        self.random_state = random_state
        self.best_model = None
        self.best_inertia = np.inf

    def fit_predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        for _ in range(self.n_sampling_runs):
            sample_idx = np.random.choice(
                len(X),
                min(self.n_samples, len(X)),
                replace=False
            )
            X_sample = X[sample_idx]

            model = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=min(1000, len(X_sample))
            ).fit(X_sample)

            inertia = model.inertia_

            if inertia < self.best_inertia:
                self.best_model = model
                self.best_inertia = inertia

        return self.best_model.predict(X)

"""Bereinigt NaN und nutzlose Kategorien---------------------------------------"""
def clean_data(df):
    cleaned_df = df.copy()
    cleaned_df.loc[:, 'year'] = pd.to_datetime(cleaned_df['update_date']).dt.year
    columns_to_keep = ['title', 'categories', 'year']
    cleaned_df = cleaned_df[columns_to_keep]
    cleaned_df.loc[:, 'title_length'] = cleaned_df['title'].str.len()
    cleaned_df.loc[:, 'title_length_scaled'] = StandardScaler().fit_transform(
        cleaned_df['title_length'].values.reshape(-1, 1)
    )
    return cleaned_df
"""Erstellt die schönen Wordclouds------------------------------------"""
def create_wordcloud(text_data, title):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOP_WORDS_SET,
        max_words=50
    ).generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout(pad=0)
    plt.savefig(f'wordcloud_{title}.png')
    plt.close()

"""Plottet die Trends----------------------------------------------------"""
def plot_cluster_trends(data, labels, min_papers=100):
    df_trends = pd.DataFrame({
        'year': data['year'],
        'cluster': labels
    })

    # Berechne absolute Zahlen pro Jahr und Cluster
    yearly_counts = pd.crosstab(df_trends['year'], df_trends['cluster'])

    # Erstelle eine neue DataFrame für Prozente mit explizitem float64 dtype: Sonst Future Warning!!!!
    yearly_percentages = pd.DataFrame(
        index=yearly_counts.index,
        columns=yearly_counts.columns,
        dtype='float64'
    )

    # Berechne und speichere Prozente
    for year in yearly_counts.index:
        year_total = yearly_counts.loc[year].sum()
        for cluster in yearly_counts.columns:
            count = yearly_counts.loc[year, cluster]
            percentage = (count / year_total) * 100
            yearly_percentages.loc[year, cluster] = percentage

    print("\nProzentuale Verteilung (Kontrolle):")
    print(yearly_percentages)
    print("\nSumme der Prozente pro Jahr (sollte 100% sein):")
    print(yearly_percentages.sum(axis=1))

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot absolute Zahlen
    for cluster in yearly_counts.columns:
        ax1.plot(yearly_counts.index, yearly_counts[cluster],
                 marker='o', label=f'Cluster {cluster}',
                 linewidth=2)

    ax1.set_title('Absolute Entwicklung der Cluster')
    ax1.set_xlabel('Jahr')
    ax1.set_ylabel('Anzahl Papers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot prozentuale Verteilung
    for cluster in yearly_percentages.columns:
        ax2.plot(yearly_percentages.index, yearly_percentages[cluster],
                 marker='o', label=f'Cluster {cluster}',
                 linewidth=2)

    ax2.set_title('Relative Entwicklung der Cluster')
    ax2.set_xlabel('Jahr')
    ax2.set_ylabel('Anteil in %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

"""Bars für die Top 10 Wörter in Prozent----------------------------------------------------"""
def plot_top_words(words, weights, cluster_id):
    """Erstellt Balkendiagramm für Top-Wörter"""
    plt.figure(figsize=(12, 6))
    y_pos = np.arange(len(words))

    # Konvertiere zu Prozent
    percentages = (weights / np.sum(weights) * 100)

    plt.barh(y_pos, percentages)
    plt.yticks(y_pos, words)
    plt.xlabel('Anteil in %')
    plt.title(f'Top 10 Wörter in Cluster {cluster_id}')

    # Füge Prozentwerte als Labels hinzu
    for i, v in enumerate(percentages):
        plt.text(v, i, f'{v:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig(f'top_words_cluster_{cluster_id}.png')
    plt.close()


def main():
#Wir wählen K manuell mit 2. Festgelgt für diese Analyse bei zwei aus Voranalyse über Sample-Code----------------
    n_clusters = 2

    print("Starting Analysis...")
    print(f"\nVerwendete Cluster-Anzahl k: {n_clusters}")

    print("\nLese Daten ein...")
    df = pd.read_json('data.json', lines=True)
    print(f"Datensätze geladen: {len(df)}")

    print("\nBereinige Daten...")
    data = clean_data(df)

    # Zeige Verteilung der Jahre
    print("\nVerteilung der Papers über die Jahre:")
    year_counts = data['year'].value_counts().sort_index()
    print(year_counts)

    print("\nSkewness vor der Transformation:")
    print(data['title_length'].skew())#Einmal um die Skewness numerisch zu visualisieren.

    print("\nErstelle Features...")
    tfidf = TfidfVectorizer(
        max_features=300,
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2)
    )
    title_tfidf = tfidf.fit_transform(data['title'])

    mlb = MultiLabelBinarizer()
    categories_encoded = mlb.fit_transform([cats.split() for cats in data['categories']])
    categories_df = pd.DataFrame(categories_encoded, columns=mlb.classes_)

    print("\nFinale Feature-Zusammensetzung:")
    print(f"TF-IDF Features: {title_tfidf.shape[1]}")
    print(f"Kategorie-Features: {categories_df.shape[1]}")
    print(f"Zusätzliche Features: 1 (title_length_scaled)")
    print(f"Gesamtzahl Features: {title_tfidf.shape[1] + categories_df.shape[1] + 1}")

    print("\nFühre Dimensionsreduktion durch...")
#Zur zeitakzeptablen Verarbeitung der Daten: SVD->PCA->t-SNE
    print("Schritt 1: TruncatedSVD für TF-IDF...")
    svd = TruncatedSVD(n_components=50)
    tfidf_svd = svd.fit_transform(title_tfidf)
    print(f"Erklärte Varianz durch SVD: {sum(svd.explained_variance_ratio_):.2%}")

    print("Schritt 2: PCA...")
    pca = PCA(n_components=30)
    tfidf_pca = pca.fit_transform(tfidf_svd)
    print(f"Erklärte Varianz durch PCA: {sum(pca.explained_variance_ratio_):.2%}")

    print("Schritt 3: t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=250,
        random_state=42
    )
    tfidf_tsne = tsne.fit_transform(tfidf_pca)

    final_features = pd.concat([
        data[['title_length_scaled']],
        categories_df,
        pd.DataFrame(tfidf_tsne, columns=['tsne_1', 'tsne_2'])
    ], axis=1)

    print("\nFühre CLARA Clustering durch...")
    clara = CLARA(
        n_clusters=n_clusters,
        n_samples=2000,
        n_sampling_runs=10,
        random_state=42
    )
    labels = clara.fit_predict(final_features)

    # Visualisierung der Cluster im t-SNE Raum: Scatterplot. 2D!
    plt.figure(figsize=(12, 8))
    plot_size = min(50000, len(tfidf_tsne))
    plot_idx = np.random.choice(len(tfidf_tsne), plot_size, replace=False)

    scatter = plt.scatter(tfidf_tsne[plot_idx, 0], tfidf_tsne[plot_idx, 1],
                          c=labels[plot_idx], cmap='viridis',
                          alpha=0.6, s=5)
    plt.colorbar(scatter)
    plt.title(f'Cluster im t-SNE Raum (Stichprobe von {plot_size} Punkten)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tsne_clusters.png', dpi=300)
    plt.close()

    # Zeitliche Entwicklung der Cluster
    plot_cluster_trends(data, labels)

    # Cluster-Analyse
    feature_names = tfidf.get_feature_names_out()
    for cluster in range(n_clusters):
        mask = labels == cluster
        cluster_size = np.sum(mask)
        cluster_percentage = (cluster_size / len(data)) * 100

        print(f"\nCluster {cluster}:")
        print(f"Größe: {cluster_size} Papers ({cluster_percentage:.1f}%)")

        if cluster_size > 0:
            # Wordcloud
            cluster_titles = ' '.join(data[mask]['title'])
            create_wordcloud(cluster_titles, f'cluster_{cluster}')

            # Top Wörter Analyse
            cluster_tfidf = title_tfidf[mask].mean(axis=0).A1
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            words = [feature_names[i] for i in top_indices]
            weights = cluster_tfidf[top_indices]

            # Plot Top Wörter
            plot_top_words(words, weights, cluster)

            total_weight = np.sum(weights)
            if total_weight > 0:
                percentages = (weights / total_weight) * 100
                print("\nTop 10 Wörter (Absolut und Prozentual):")
                print("Wort: absoluter Wert (prozentualer Anteil)")
                print("-" * 45)
                for word, weight, percentage in zip(words, weights, percentages):
                    print(f"{word:20}: {weight:.4f} ({percentage:.2f}%)")

            # Zeitliche Analyse
            years = data[mask]['year']
            print(f"\nZeitspanne: {years.min()} - {years.max()}")
            print(f"Durchschnittsjahr: {years.mean():.1f}")

            # Kategorie-Analyse
            print("\nTop 5 Kategorien:")
            cluster_categories = data[mask]['categories'].str.split().explode()
            top_categories = cluster_categories.value_counts().head(5)
            for cat, count in top_categories.items():
                percentage = (count / cluster_size) * 100
                print(f"{cat}: {count} Papers ({percentage:.1f}%)")

    print("\nAnalyse abgeschlossen.")

if __name__ == "__main__":
    main()