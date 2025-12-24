import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy.stats import boxcox
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class DataCleaner:
    """
    Cleaning and preprocessing retail transaction data

    This class handles data loading, cleaning operations, and basic exploratory
    data analysis for online retail datasets.
    """

    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to raw data file
        """
        self.data_path = data_path
        self.df = None
        self.df_ul = None
        self.rfm_data = None

    def load_data(self):
        """
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        dtype = dict(
            InvoiceNo=np.object_,
            StockCode=np.object_,
            Description=np.object_,
            Quantity=np.int64,
            UnitPrice=np.float64,
            CustomerID=np.object_,
            Country=np.object_,
        )
        
        self.df = pd.read_csv(
            self.data_path,
            encoding="ISO-8859-1",
            parse_dates=["InvoiceDate"],
            dtype=dtype,
        )

        self.df["CustomerID"] = (
            self.df["CustomerID"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
        )
        
        print(f"Data size: {self.df.shape}")
        print(f"Number of Records: {len(self.df):,}")
        
        return self.df
    
    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing or UK customers.add()
        
        Returns:
            pd.DataFrame: Cleaned UK dataset
        """
        # Add TotalPrice column
        self.df["TotalPrice"] = self.df["Quantity"] * self.df["UnitPrice"]
        # Remove cancelled invoices (starting with 'C')
        self.df = self.df[~self.df["InvoiceNo"].astype(str).str.startswith("C")]
        # Focus on UK customers
        self.df_uk = self.df[self.df["Country"] == "United Kingdom"].copy()
        
        # Remove invalid quantiy or unit price
        self.df_uk = self.df_uk[
            (self.df_uk["Quantity"] > 0) & (self.df_uk["UnitPrice"] > 0)
        ]
        
        return self.df_uk
    
    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        self.df_uk["DayOfWeek"] = self.df_uk["InvoiceDate"].dt.dayofweek
        self.df_uk["HourOfDay"] = self.df_uk["InvoiceDate"].dt.hour
        
    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics.
        
        Returns:
            pd.DataFrameL: RFM data for each customer
        """
        snapshot_date = self.df_uk["InvoiceDate"].max() + pd.Timedelta(days=1)
        
        self.rfm_data = self.df_uk.groupby("CustomerID").agg(
            {
                "InvoiceDate": lambda x: (snapshot_date - x.max()).days, # Recency
                "InvoiceNo": lambda x: len(x.unique()), # Frequency
                "TotalPrice": lambda x: x.sum(), # Monetary       
            }
        )
        
        self.rfm_data.columns = ["Recency", "Frequency", "Monetary"]
        return self.rfm_data
    
    def save_cleaned_data(self, output_dir="../data/processed"):
        """
        Save cleaned data to specified directory.
        
        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        self.df_uk.to_csv(f"{output_dir}/cleaned_uk_data.csv", index=False)
        print(f"Cleaned data saved to {output_dir}/cleaned_uk_data.csv")


class FeatureEngineer:
    """
    A class for creating customer-level features from transaction data.

    This class aggregates transaction-level dta into customer-level features
    for clustering analysis.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.customer_features_scaled = None
    
        self.feature_customer = [
            "Sum_Quantity",
            "Mean_UnitPrice",
            "Mean_TotalPrice",
            "Sum_TotalPrice",
            "Count_Invoice",
            "Count_Stock",
            "Mean_InvoiceCountPerStock",
            "Mean_StockCountPerInvoice",
            "Mean_UnitPriceMeanPerInvoice",
            "Mean_QuantitySumPerInvoice",
            "Mean_TotalPriceMeanPerInvoice",
            "Mean_TotalPriceSumPerInvoice",
            "Mean_UnitPriceMeanPerStock",
            "Mean_QuantitySumPerStock",
            "Mean_TotalPriceMeanPerStock",
            "Mean_TotalPriceSumPerStock",
        ]
        
        self.feature_customer2 = ["CustomerID"] + self.feature_customer
        
    def load_data(self):
        """
        Load cleaned data and prepare for feature engineering.

        Returns:
            pd.DataFrame: Loaded cleaned data
        """
        self.df = pd.read_csv(self.data_path)
        self.df["InvoiceDate"] = pd.to_datetime(self.df["InvoiceDate"])

        print(f"Data shape: {self.df.shape}")
        return self.df
    
    def create_customer_features(self):
        """
        Create customer-level aggregated features.
        
        Returns:
            pd.DataFrame: Customer feature dataframe
        """
        num_customers = self.df["CustomerID"].nunique()
        self.customer_features = pd.DataFrame(
            data=np.zeros((num_customers, len(self.feature_customer2)), dtype=float),
            columns=self.feature_customer2,
        )
        
        self.customer_features["CustomerID"] = self.customer_features["CustomerID"].astype("object")
        
        print("Calculating customer features...")
        
        for i, (customer_id, value) in enumerate(self.df.groupby("CustomerID")):
            self.customer_features.iat[i, 0] = customer_id
            self.customer_features.iat[i, 1] = value.Quantity.sum()
            self.customer_features.iat[i, 2] = value.UnitPrice.mean()
            self.customer_features.iat[i, 3] = value.TotalPrice.mean()
            self.customer_features.iat[i, 4] = value.TotalPrice.sum()
            self.customer_features.iat[i, 5] = value.InvoiceNo.nunique()
            self.customer_features.iat[i, 6] = value.StockCode.nunique()
            
            self.customer_features.iat[i, 7] = value.groupby("StockCode").size().mean()
            self.customer_features.iat[i, 8] = value.groupby("InvoiceNo").size().mean()
            self.customer_features.iat[i, 9] = (
                value.groupby("InvoiceNo")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 10] = (
                value.groupby("InvoiceNo")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 11] = (
                value.groupby("InvoiceNo")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 12] = (
                value.groupby("InvoiceNo")["TotalPrice"].sum().mean()
            )
            self.customer_features.iat[i, 13] = (
                value.groupby("StockCode")["UnitPrice"].mean().mean()
            )
            self.customer_features.iat[i, 14] = (
                value.groupby("StockCode")["Quantity"].sum().mean()
            )
            self.customer_features.iat[i, 15] = (
                value.groupby("StockCode")["TotalPrice"].mean().mean()
            )
            self.customer_features.iat[i, 16] = (
                value.groupby("StockCode")["TotalPrice"].sum().mean()
            )
            
            if (i + 1) % 500 == 0:
                print(f"Processed {i + 1}/{num_customers} customers...")
                
        print("✓ Customer features calculated!")
        return self.customer_features
        
        
    def transform_features(self):
        """
        Apply Box-Cox transformation to normalize feature distributions.
        
        Returns:
            pd.DataFrame: Transformed features
        """
        customer_features_indexed = self.customer_features.set_index("CustomerID")
        
        feature_values = customer_features_indexed.values + 1
        
        self.customer_features_transformed = customer_features_indexed.copy()
        
        print("Applying Box-Cox transformation...")
        for i, feature in enumerate(self.feature_customer):
            transformed, lambda_param = boxcox(feature_values[:, i])
            self.customer_features_transformed.iloc[:, i] = transformed
            
        print("✓ Box-Cox transformation applied!")
        return self.customer_features_transformed
    
    def scale_features(self):
        """
        Apply standardization to features.
        
        Returns:
            pd.DataFrame: Scaled features
        """
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.customer_features_transformed)
        
        self.customer_features_scaled = pd.DataFrame(
            features_scaled, 
            columns=self.feature_customer,
            index=self.customer_features_transformed.index,
        )
        
        print("✓ Features scaled!")
        return self.customer_features_scaled
    
    def plot_features_boxplots(self, transformed=False, save_path=None):
        """
        Box plots for all features.
        
        Args:
            transformed (bool): True to plot transformed features, False for original
            save_path (str): Path to save the plot image
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")            
                title = "Box Plots of Customer Features (Original)"
            else:
                print(
                    "Error: No feature data. Please run create_customer_features() first."
                )
                return
        
        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))
            
            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.boxplot(data.iloc[:, i] if transformed else data[feature])
                plt.title(feature, fontsize=10)
                plt.xticks([])
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Box plots saved to {save_path}")
                
            plt.show()
            
    def plot_features_histograms(self, transformed=False, save_path=None):
        """
        Histograms for all features.
        
        Args:
            transformed (bool): True to plot transformed features, False for original
            save_path (str): Path to save the plot image
        """
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
            title = "Histograms of Customer Features (Transformed)"
        else:
            if self.customer_features is not None:
                data = self.customer_features.set_index("CustomerID")
                title = "Histograms of Customer Features (Original)"
            else:
                print(
                    "Error: No feature data. Please run create_customer_features() first."
                )
                return
        
        with sns.plotting_context(context="notebook"):
            plt.figure(figsize=(15, 15))
            
            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i + 1)
                plt.hist(
                    data.iloc[:, i] if transformed else data[feature],
                    bins=30,
                    alpha=0.9,
                )
                plt.title(feature, fontsize=10)
                plt.ylabel("Frequency", fontsize=8)
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches="tight")
                print(f"Histograms saved to {save_path}")
            plt.show()
            
    def save_features(self, output_dir="../data/processed"):
        """
        Save all processed features.
        
        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        customer_features_indexed = self.customer_features.set_index("CustomerID")
        customer_features_indexed.to_csv(f"{output_dir}/customer_features.csv")
        
        self.customer_features_transformed.to_csv(
            f"{output_dir}/customer_features_transformed.csv"
        )
        
        self.customer_features_scaled.to_csv(
            f"{output_dir}/customer_features_scaled.csv"
        )
        
        print(f"✓ Features saved to {output_dir}/")
    
    
class ClusterAnalyzer:
    """
    A class for performing clustering analysis and visualizations.
    
    This class handles PCA, optimal cluster determination, K-means clustering
    and cluster visualization and interpretation.
    """
    FEATURE_NAMES_VN = {
        "Sum_Quantity": "Total quantity purchased",
        "Mean_UnitPrice": "Average price",
        "Mean_TotalPrice": "Average transaction value",
        "Sum_TotalPrice": "Total spending",
        "Count_Invoice": "Number of purchases",
        "Count_Stock": "Number of unique products",
        "Mean_InvoiceCountPerStock": "Purchase frequency per product",
        "Mean_StockCountPerInvoice": "Products per transaction",
        "Mean_UnitPriceMeanPerInvoice": "Average price per transaction",
        "Mean_QuantitySumPerInvoice": "Quantity per transaction",
        "Mean_TotalPriceMeanPerInvoice": "Average value per transaction",
        "Mean_TotalPriceSumPerInvoice": "Total value per transaction",
        "Mean_UnitPriceMeanPerStock": "Average price per product",
        "Mean_QuantitySumPerStock": "Average quantity per product",
        "Mean_TotalPriceMeanPerStock": "Average value per product",
        "Mean_TotalPriceSumPerStock": "Total value per product",
    }
     
    def __init__(self, scaled_features_path, original_features_path):
        """
        Args:
            scaled_features_path (str): Path to scaled features CSV
            original_features_path (str): Path to original features CSV
        """
        self.scaled_features_path = scaled_features_path
        self.original_features_path = original_features_path
        self.df_scaled = None 
        self.df_original = None
        self.df_pca = None 
        self.optimal_clusters = {}
        self.cluster_results = {}
        self.surrogate_models = {}
        self.shap_results = {}
        
    def load_data(self):
        """
        Load scaled and original features data.

        Returns:
            tuple: (scaled_features_df, original_features_df)
        """
        self.df_scaled = pd.read_csv(self.scaled_features_path, index_col=0)
        self.df_original = pd.read_csv(self.original_features_path, index_col=0)
        
        print(f"Number of customers: {self.df_scaled.shape[0]}")
        print(f"Number of features: {self.df_scaled.shape[1]}")
        
        return self.df_scaled, self.df_original
        
    def apply_pca(self, n_components=None):
        """
        Apply PCA.
        
        Args:
            n_components (int): Number of PCA components to keep
        Returns:
            pd.DataFrame: PCA-transformed data
        """
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(self.df_scaled)
        
        pca_columns = [f"PC{i+1}" for i in range(pca_features.shape[1])]
        self.df_pca = pd.DataFrame(
            pca_features, columns=pca_columns, index=self.df_scaled.index
        )
         
        print(f"PCA shape: {self.df_pca.shape}")
        return self.df_pca

    def plot_pca_variance(self):
        """
        Plot explained variance ratio by PCA components.
        """
        plt.figure(figsize=(12, 6))
        
        plt.bar(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            self.pca.explained_variance_ratio_,
            alpha=0.7,
            label="Individual explained variance",
        )
        
        plt.step(
            range(1, len(self.pca.explained_variance_ratio_) + 1),
            np.cumsum(self.pca.explained_variance_ratio_),
            where="mid",
            label="Cumulative explained variance",
            color="red",
            linewidth=2,
        )
        
        plt.axhline(y=0.8, color="green", linestyle="--", label="80% variance")
        plt.axhline(y=0.9, color="orange", linestyle="--", label="90% variance")
        
        plt.xlabel("Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("PCA analysis - Explained variance")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print("\nCumulative variance:")
        for i in range(min(5, len(self.pca.explained_variance_ratio_))):
            cumsum = np.sum(self.pca.explained_variance_ratio_[: i + 1])
            print(f"PC1-PC{i+1}: {cumsum:.2%}")

    def find_optimal_clusters(self, k_range=range(2, 11)):
        """
        Find optimal number of clusters using multiple methods.
        
        Args:
            k_range (range): Range of k values to test.
            
        Returns:
            dict: Results from different methods
        """
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_scaled)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_scaled, labels))

        self.optimal_clusters = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "best_k_silhouette": list(k_range)[np.argmax(silhouette_scores)],
        }

        return self.optimal_clusters
    
    def plot_optimal_clusters(self):
        """
        Plot Elbow method and Silhouette scores for cluster selection.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        axes[0].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["inertias"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="blue"
        )
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(
            self.optimal_clusters["k_range"],
            self.optimal_clusters["silhouette_scores"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="green"
        )
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Score Method")
        axes[1].grid(True, alpha=0.3)
        
        best_k = self.optimal_clusters["best_k_silhouette"]
        best_score = max(self.optimal_clusters["silhouette_scores"])
        axes[1].scatter(best_k, best_score, color="red", alpha=0.5, s=200)
        axes[1].annotate(
            f"Best k={best_k}",
            xy=(best_k, best_score),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )
        
        plt.tight_layout()
        plt.show()
        
        print(f"Silhouette method suggests optimal k = {best_k} (score: {best_score:.3f})")
        
    def apply_kmeans(self, k_values=[3, 4]):
        """
        Apply K-means clustering with different k values.

        Args:
            k_values (list): List of k values to apply

        Returns:
            dict: Clustering results for each k
        """
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.df_scaled)
            
            cluster_col = f"Cluster_{k}"
            self.df_scaled[cluster_col] = clusters
            self.df_pca[cluster_col] = clusters
            self.df_original[cluster_col] = clusters

            self.cluster_results[k] = {
                "labels": clusters,
                "sizes": pd.Series(clusters).value_counts().sort_index(),
                "means": self.df_original.groupby(cluster_col).mean(),
            }
            
            print(f"Cluster size (k={k}):")
            print(self.cluster_results[k]["sizes"])
            
        return self.cluster_results

    def plot_clusters_pca(self, k_values=[3, 4]):
        """
        Visualize clusters in PCA space.

        Args:
            k_values (list): List of k values to visualize
        """
        fig, axes = plt.subplots(1, len(k_values), figsize=(16, 6))
        if len(k_values) == 1:
            axes = [axes]

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            scatter = axes[i].scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )
            axes[i].set_xlabel("PC1")
            axes[i].set_ylabel("PC2")
            axes[i].set_title(f"Kmeans cluster (k={k})")
            plt.colorbar(scatter, ax=axes[i], label="Cluster")

        plt.tight_layout()
        plt.show()

    def plot_clusters_pca_3d(self, k_values=[3, 4]):
        """
        Visualize clusters in 3D PCA space.

        Args:
            k_values (list): List of k values to visualize
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 6))

        for i, k in enumerate(k_values):
            cluster_col = f"Cluster_{k}"
            ax = fig.add_subplot(1, len(k_values), i + 1, projection="3d")

            scatter = ax.scatter(
                self.df_pca["PC1"],
                self.df_pca["PC2"],
                self.df_pca["PC3"],
                c=self.df_pca[cluster_col],
                cmap="viridis",
                alpha=0.6,
                s=50,
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"Kmeans cluster 3D (k={k})")

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label="Cluster", shrink=0.5)

        plt.tight_layout()
        plt.show()
        
    def create_radar_chart(self, k, cluster_names=None):
        """
        Create professional radar chart for cluster analysis.

        Args:
            k (int): Number of clusters
            cluster_names (list): Custom names for clusters
        """
        cluster_means = self.cluster_results[k]["means"]

        # Select key features for the radar chart
        important_features = {
            "Sum_Quantity": "Purchasing quantity",
            "Sum_TotalPrice": "Total spending",
            "Mean_UnitPrice": "Preferred price level",
            "Count_Invoice": "Purchase frequency",
            "Count_Stock": "Product diversity",
            "Mean_TotalPriceSumPerInvoice": "Value per transaction",
        }

        # Filter and normalize data
        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Global normalization
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Replace with readable labels
        data_normalized.columns = [
            important_features[col] for col in data_normalized.columns
        ]

        # Setup radar chart
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Colors
        colors = (
            ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
            if k == 4
            else ["#E74C3C", "#2ECC71", "#3498DB"]
        )
        if not cluster_names:
            cluster_names = [f"Cluster {i}" for i in range(k)]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection="polar"))

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"
            )

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=4,
                label=cluster_name,
                color=color,
                markersize=10,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.15, color=color)

        # Styling
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, weight="bold", color="#2C3E50")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(
            ["20%", "40%", "60%", "80%", "100%"], size=10, color="#7F8C8D"
        )
        ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
        ax.set_facecolor("#FAFAFA")

        ax.set_title(
            f"Customer segment analysis (K={k})",
            size=16,
            weight="bold",
            pad=30,
            color="#2C3E50",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

        plt.tight_layout()
        plt.show()
        
    def create_individual_radar_plots(self, k, cluster_names=None):
        """
        Create an individual radar plot for each cluster.

        Args:
            k (int): Number of clusters
            cluster_names (list): Custom names for clusters
        """
        cluster_means = self.cluster_results[k]["means"]

        # Select key features
        important_features = {
            "Sum_Quantity": "Purchase volume",
            "Sum_TotalPrice": "Total spending",
            "Mean_UnitPrice": "Preferred price level",
            "Count_Invoice": "Purchase frequency",
            "Count_Stock": "Product variety",
            "Mean_TotalPriceSumPerInvoice": "Value per transaction",
            "Mean_TotalPriceMeanPerStock": "Spending per product",
            "Mean_StockCountPerInvoice": "Products per transaction",
        }

        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Normalize data
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Replace labels with readable text
        data_normalized.columns = [
            important_features[col] for col in data_normalized.columns
        ]

        # Setup angles
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Professional color palette
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        if not cluster_names:
            cluster_names = [f"Cluster {i}" for i in range(k)]

        # Create subplots for each cluster with an optimal layout
        if k == 4:
            # 2x2 layout for k=4
            nrows, ncols = 2, 2
            figsize = (12, 10)
        else:
            # Single-row layout for other values of k
            nrows, ncols = 1, k
            figsize = (5 * k, 5)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, subplot_kw=dict(projection="polar")
        )

        # Ensure axes is always a 2D array for consistent handling
        if k == 1:
            axes = np.array([[axes]])
        elif k == 4:
            # axes is already a 2D array (2x2)
            pass
        else:
            # Reshape to 2D array for consistency
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            # Compute position within the 2D grid
            if k == 4:
                row_idx, col_idx = idx // 2, idx % 2
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[0, idx] if len(axes.shape) == 2 else axes[idx]

            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"
            )

            # Draw radar chart
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=3,
                label=cluster_name,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.25, color=color)

            # Styling
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11, weight="bold", color="#2C3E50")
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(
                ["20%", "40%", "60%", "80%", "100%"], size=9, color="#7F8C8D"
            )
            ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
            ax.set_facecolor("#FAFAFA")

            # Title for each subplot
            ax.set_title(
                f"{cluster_name}\n({cluster_means.index[idx]})",
                size=13,
                weight="bold",
                pad=20,
                color=color,
            )

        plt.suptitle(
            f"Detailed analysis for each cluster (K={k})", size=16, weight="bold", y=1.05
        )
        plt.tight_layout()
        plt.show()

    def train_surrogate_model(self, k):
        """
        Train a RandomForest classifier to mimic the KMeans assignments.
        The surrogate model is used for SHAP-based explanations.
        
        Args:
            k (int): Number of clusters
            
        Returns:
            dict: Training results including model and metrics
        """
        if k not in self.cluster_results:
            raise ValueError(f"Cluster results for k={k} not found. Run apply_kmeans first.")
        
        # Use all columns that are not cluster labels
        feature_cols = [col for col in self.df_scaled.columns if not col.startswith('Cluster_')]
        X = self.df_scaled[feature_cols].values
        y = self.cluster_results[k]['labels']
        
        # Train RandomForest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        
        # Predict
        y_pred = rf_model.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Store results
        self.surrogate_models[k] = {
            'model': rf_model,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_cols
        }
        
        # Report results
        print(f"=== TRAINED SURROGATE MODEL (k={k}) ===")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print(f"\nThe model can predict the clusters {'ACCURATELY' if accuracy >= 0.95 else 'RELIABLY'}.")
        
        return self.surrogate_models[k]
    
    def calculate_shap_values(self, k):
        """
        Compute SHAP values to explain cluster assignments using the full dataset.
        
        Args:
            k (int): Number of clusters
            
        Returns:
            dict: SHAP explainer and values
        """
        if k not in self.surrogate_models:
            raise ValueError(f"Surrogate model for k={k} not found. Please run train_surrogate_model first.")
        
        # Retrieve model and feature set
        rf_model = self.surrogate_models[k]['model']
        feature_cols = self.surrogate_models[k]['feature_names']
        X = self.df_scaled[feature_cols].values
        
        # Create SHAP explainer with the full dataset as background for accuracy
        print(f"Computing SHAP values for {len(X):,} customers...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values_raw = explainer.shap_values(X)
        
        # Convert to list format for multi-class outputs
        # Shape: (n_samples, n_features, n_classes) -> list (n_samples, n_features)
        if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
        else:
            shap_values = shap_values_raw
        
        # Store results
        self.shap_results[k] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_cols,
            'X': X
        }
        
        print(f"Done! SHAP values: {len(shap_values)} clusters, each cluster shape: {shap_values[0].shape}")
        return self.shap_results[k]
    
    def plot_shap_summary(self, k, cluster_id=None):
        """
        Plot SHAP summary (beeswarm) for cluster analysis.
        
        Args:
            k (int): Number of clusters
            cluster_id (int, optional): Specific cluster to visualize. If None, shows all.
        """
        if k not in self.shap_results:
            raise ValueError(f"SHAP values for k={k} not found. Please run calculate_shap_values first.")
        
        shap_values = self.shap_results[k]['shap_values']
        X = self.shap_results[k]['X']
        feature_names = self.shap_results[k]['feature_names']
    
        for i in range(k):
            shap.summary_plot(
                shap_values[i],
                X,
                feature_names=feature_names,
                max_display=3,
                show=True
            )

    def save_clusters(self, output_dir="../data/processed"):
        """
        Save cluster assignments.

        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        for k in self.cluster_results.keys():
            cluster_col = f"Cluster_{k}"
            cluster_output = self.df_original[[cluster_col]].copy()
            cluster_output.columns = ["Cluster"]
            cluster_output = cluster_output.reset_index()
            cluster_output = cluster_output.sort_values(["Cluster", "CustomerID"])

            cluster_output.to_csv(
                f"{output_dir}/customer_clusters_k{k}.csv", index=False
            )
            print(
                f"Saved clustering results for k={k}: {output_dir}/customer_clusters_k{k}.csv"
            )
        
        
class DataVisualizer():
    """
    Creating visualizations for customer segmentation analysis.

    This class provides methods for plotting various aspects of the data 
    including temporal patterns, customer behvior, and cluster analyis.
    """

    def __init__(self):
        """Initialize the DataVisualize with plotting  settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Daily Revenue")
        plt.xlabel("Day")
        plt.ylabel("Revenue (GBP)")
        plt.tight_layout()
        plt.show()
        
        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Revenue (GPB)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.
        
        Args:
            df (pd.DataFrame): DataFrame with time features
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.xlabel("Shopping activity by day time")
        plt.ylabel("Time in day")
        plt.tight_layout()
        plt.show()
        
    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe
            top_n (int): Number of top products to show
        """
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} products by sales volume")
        plt.xlabel("Quantity")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} by sales volume")
        plt.xlabel("Revenue (BGP)")
        plt.tight_layout()
        plt.show()
        
    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.
        
        Args:
            df (pd.DataFrame): Transaction dataframe
        """
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Distribution of Transactions per Customer")
        plt.xlabel("Number of transactions")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Distribution of Spend per Customer")
        plt.xlabel("Total Spend (GBP)")
        plt.ylabel("Number of Customers")
        plt.tight_layout()
        plt.show()
        
    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM score distributions.
        
        Args:
            rfm_data (pd.DataFrame): DataFrame with RFM scores
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Recency Distribution")
        axes[0].set_xlabel("Recency (days)")
        
        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Frequency Distribution")
        axes[1].set_xlabel("Frequency (number of purchases)")
        
        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data["Monetary"][monetary_filter], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Monetary Distribution")
        axes[2].set_xlabel("Monetary (GBP)")
        plt.tight_layout()
        plt.show()