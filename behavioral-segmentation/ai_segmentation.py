import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import umap
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des modèles
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CLUSTERING_ALGORITHMS = ['kmeans', 'dbscan']

@dataclass
class UserSegment:
    """Structure pour représenter un segment d'utilisateur"""
    segment_id: int
    segment_name: str
    user_ids: List[str]
    size: int
    characteristics: Dict[str, Any]
    representative_queries: List[str]
    confidence_score: float

@dataclass
class SegmentationResults:
    """Résultats complets de la segmentation"""
    segments: List[UserSegment]
    user_assignments: Dict[str, int]
    cluster_centers: np.ndarray
    quality_metrics: Dict[str, float]
    feature_importance: Dict[str, float]

class AISegmentationEngine:
    """
    Moteur de segmentation IA utilisant des embeddings et du clustering
    pour identifier des groupes d'utilisateurs homogènes
    """
    
    def __init__(self, 
                 embedding_model: str = EMBEDDING_MODEL,
                 random_state: int = 42,
                 cache_embeddings: bool = True):
        """
        Initialise le moteur de segmentation IA
        
        Args:
            embedding_model: Modèle pour les embeddings de texte
            random_state: Seed pour la reproductibilité
            cache_embeddings: Cache les embeddings pour éviter le recalcul
        """
        self.embedding_model_name = embedding_model
        self.random_state = random_state
        self.cache_embeddings = cache_embeddings
        self.embedding_model = None
        self.scaler = StandardScaler()
        self.embeddings_cache = {}
        
        # Initialise le modèle d'embedding
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Charge le modèle d'embedding"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print(f"🔄 Chargement du modèle d'embedding: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            print("✅ Modèle d'embedding chargé avec succès")
            
        except ImportError:
            logger.error("❌ sentence-transformers non installé")
            print("💡 Installation: pip install sentence-transformers")
            # Fallback vers une simulation d'embeddings
            self.embedding_model = None
            print("⚠️  Utilisation d'embeddings simulés pour la démonstration")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            self.embedding_model = None
            print("⚠️  Utilisation d'embeddings simulés pour la démonstration")
    
    def generate_search_embeddings(self, user_queries: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Génère les embeddings pour les requêtes de recherche de chaque utilisateur
        
        Args:
            user_queries: Dictionnaire {user_id: [liste des requêtes]}
            
        Returns:
            Dictionnaire {user_id: embedding_vector}
        """
        print("🔤 Génération des embeddings de recherche...")
        
        user_embeddings = {}
        
        for user_id, queries in user_queries.items():
            if not queries:
                continue
            
            # Vérification du cache
            cache_key = f"{user_id}_{hash(tuple(sorted(queries)))}"
            if self.cache_embeddings and cache_key in self.embeddings_cache:
                user_embeddings[user_id] = self.embeddings_cache[cache_key]
                continue
            
            try:
                if self.embedding_model is not None:
                    # Utilise le vrai modèle d'embedding
                    query_embeddings = self.embedding_model.encode(queries)
                    
                    # Agrège les embeddings (moyenne pondérée)
                    if len(query_embeddings) == 1:
                        user_embedding = query_embeddings[0]
                    else:
                        # Moyenne avec pondération basée sur la fréquence
                        weights = np.ones(len(query_embeddings)) / len(query_embeddings)
                        user_embedding = np.average(query_embeddings, axis=0, weights=weights)
                else:
                    # Embedding simulé basé sur les mots-clés
                    user_embedding = self._simulate_embedding(queries)
                
                user_embeddings[user_id] = user_embedding
                
                # Cache l'embedding
                if self.cache_embeddings:
                    self.embeddings_cache[cache_key] = user_embedding
                    
            except Exception as e:
                logger.warning(f"❌ Erreur embedding pour {user_id}: {e}")
                # Embedding par défaut
                user_embeddings[user_id] = np.random.normal(0, 0.1, 384)  # Dimension par défaut
        
        print(f"✅ Embeddings générés pour {len(user_embeddings)} utilisateurs")
        return user_embeddings
    
    def _simulate_embedding(self, queries: List[str], dim: int = 384) -> np.ndarray:
        """
        Simule un embedding basé sur les mots-clés des requêtes
        Utilisé quand sentence-transformers n'est pas disponible
        """
        # Mots-clés par catégorie avec leurs vecteurs caractéristiques
        category_keywords = {
            'tech': ['laptop', 'phone', 'computer', 'gaming', 'tech', 'wireless', 'bluetooth', 'smart'],
            'fashion': ['dress', 'shoes', 'fashion', 'clothing', 'style', 'designer', 'luxury'],
            'home': ['furniture', 'home', 'kitchen', 'decor', 'appliance', 'bedding'],
            'sports': ['fitness', 'sports', 'gym', 'running', 'exercise', 'yoga'],
            'budget': ['cheap', 'discount', 'sale', 'affordable', 'budget', 'best price']
        }
        
        # Vecteurs de base pour chaque catégorie
        base_vectors = {
            'tech': np.random.normal(1.0, 0.2, dim),
            'fashion': np.random.normal(-0.5, 0.2, dim),
            'home': np.random.normal(0.3, 0.2, dim),
            'sports': np.random.normal(-1.0, 0.2, dim),
            'budget': np.random.normal(0.8, 0.3, dim)
        }
        
        # Calcule le score pour chaque catégorie
        category_scores = defaultdict(float)
        
        for query in queries:
            query_lower = query.lower()
            for category, keywords in category_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        category_scores[category] += 1
        
        # Crée l'embedding comme combinaison pondérée des vecteurs de base
        if not category_scores:
            return np.random.normal(0, 0.1, dim)
        
        total_score = sum(category_scores.values())
        embedding = np.zeros(dim)
        
        for category, score in category_scores.items():
            weight = score / total_score
            embedding += weight * base_vectors[category]
        
        # Normalise l'embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def create_user_profiles(self, 
                           ml_data: pd.DataFrame, 
                           user_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """
        Combine les embeddings et les features comportementales pour créer des profils utilisateur
        
        Args:
            ml_data: DataFrame avec les features ML
            user_embeddings: Embeddings des requêtes utilisateur
            
        Returns:
            Tuple (profiles_matrix, user_ids)
        """
        print("👤 Création des profils utilisateur...")
        
        # Aligne les données
        common_users = set(ml_data['user_id']).intersection(set(user_embeddings.keys()))
        print(f"📊 {len(common_users)} utilisateurs avec données complètes")
        
        if len(common_users) == 0:
            raise ValueError("Aucun utilisateur avec données complètes")
        
        # Ordonne les utilisateurs
        user_ids = sorted(list(common_users))
        
        # Prépare les features comportementales
        behavioral_features = []
        embedding_features = []
        
        for user_id in user_ids:
            # Features comportementales (normalisées)
            user_ml_data = ml_data[ml_data['user_id'] == user_id].iloc[0]
            
            # Sélectionne les features numériques importantes
            feature_columns = [
                'search_frequency_normalized',
                'avg_session_duration_normalized', 
                'click_through_rate_normalized',
                'conversion_rate_normalized',
                'engagement_score_normalized',
                'activity_intensity_normalized',
                'prefers_electronics',
                'prefers_clothing',
                'prefers_home',
                'prefers_sports',
                'price_sensitivity_high',
                'price_sensitivity_medium'
            ]
            
            # Utilise les features disponibles
            available_features = [col for col in feature_columns if col in user_ml_data.index]
            behavior_vector = user_ml_data[available_features].values
            
            behavioral_features.append(behavior_vector)
            
            # Embedding des requêtes
            embedding_vector = user_embeddings[user_id]
            embedding_features.append(embedding_vector)
        
        # Convertit en arrays numpy
        behavioral_matrix = np.array(behavioral_features)
        embedding_matrix = np.array(embedding_features)
        
        # Normalise les embeddings si nécessaire
        if embedding_matrix.std() > 1:
            embedding_matrix = StandardScaler().fit_transform(embedding_matrix)
        
        # Combine les features (pondération)
        behavior_weight = 0.3
        embedding_weight = 0.7
        
        # Ajuste les dimensions si nécessaire
        if behavioral_matrix.shape[1] != embedding_matrix.shape[1]:
            # Réduit les embeddings à la dimension des features comportementales
            if embedding_matrix.shape[1] > behavioral_matrix.shape[1]:
                pca = PCA(n_components=behavioral_matrix.shape[1])
                embedding_matrix = pca.fit_transform(embedding_matrix)
            else:
                # Étend les features comportementales
                pca = PCA(n_components=embedding_matrix.shape[1])
                behavioral_matrix = pca.fit_transform(
                    np.hstack([behavioral_matrix, 
                              np.random.normal(0, 0.1, 
                                             (behavioral_matrix.shape[0], 
                                              embedding_matrix.shape[1] - behavioral_matrix.shape[1]))])
                )
        
        # Combine les matrices
        combined_profiles = (
            behavior_weight * behavioral_matrix + 
            embedding_weight * embedding_matrix
        )
        
        print(f"✅ Profils créés: {combined_profiles.shape[0]} utilisateurs, {combined_profiles.shape[1]} dimensions")
        
        return combined_profiles, user_ids
    
    def determine_optimal_clusters(self, 
                                 user_profiles: np.ndarray, 
                                 max_clusters: int = 10,
                                 min_clusters: int = 2) -> int:
        """
        Détermine le nombre optimal de clusters
        
        Args:
            user_profiles: Matrice des profils utilisateur
            max_clusters: Nombre maximum de clusters à tester
            min_clusters: Nombre minimum de clusters
            
        Returns:
            Nombre optimal de clusters
        """
        print("🔍 Détermination du nombre optimal de clusters...")
        
        if len(user_profiles) < min_clusters:
            return min(len(user_profiles), 2)
        
        max_clusters = min(max_clusters, len(user_profiles) - 1)
        
        # Méthodes d'évaluation
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        cluster_range = range(min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            try:
                # K-means
                kmeans = KMeans(n_clusters=n_clusters, 
                              random_state=self.random_state,
                              n_init=10)
                cluster_labels = kmeans.fit_predict(user_profiles)
                
                # Métriques
                inertia = kmeans.inertia_
                silhouette = silhouette_score(user_profiles, cluster_labels)
                calinski = calinski_harabasz_score(user_profiles, cluster_labels)
                
                inertias.append(inertia)
                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski)
                
                print(f"  {n_clusters} clusters: silhouette={silhouette:.3f}, calinski={calinski:.1f}")
                
            except Exception as e:
                logger.warning(f"Erreur pour {n_clusters} clusters: {e}")
                inertias.append(float('inf'))
                silhouette_scores.append(-1)
                calinski_scores.append(0)
        
        # Méthode du coude pour l'inertie
        def find_elbow(values):
            if len(values) < 3:
                return 0
            
            # Calcule les dérivées secondes
            diffs = np.diff(values)
            second_diffs = np.diff(diffs)
            
            if len(second_diffs) == 0:
                return 0
            
            return np.argmax(second_diffs) + min_clusters
        
        # Choix basé sur plusieurs critères
        elbow_point = find_elbow(inertias)
        best_silhouette_idx = np.argmax(silhouette_scores)
        best_calinski_idx = np.argmax(calinski_scores)
        
        # Score composite
        scores = []
        for i, n_clusters in enumerate(cluster_range):
            # Normalise les métriques
            sil_norm = silhouette_scores[i] if silhouette_scores[i] > 0 else 0
            cal_norm = calinski_scores[i] / max(calinski_scores) if max(calinski_scores) > 0 else 0
            
            # Score composite (privilégie silhouette)
            composite_score = 0.6 * sil_norm + 0.4 * cal_norm
            scores.append(composite_score)
        
        optimal_idx = np.argmax(scores)
        optimal_clusters = cluster_range[optimal_idx]
        
        print(f"✅ Nombre optimal de clusters: {optimal_clusters}")
        print(f"   Silhouette score: {silhouette_scores[optimal_idx]:.3f}")
        print(f"   Calinski-Harabasz score: {calinski_scores[optimal_idx]:.1f}")
        
        return optimal_clusters
    
    def perform_clustering(self, 
                         user_profiles: np.ndarray, 
                         user_ids: List[str],
                         n_clusters: Optional[int] = None,
                         algorithm: str = 'kmeans') -> SegmentationResults:
        """
        Effectue le clustering des utilisateurs
        
        Args:
            user_profiles: Matrice des profils utilisateur
            user_ids: Liste des IDs utilisateur
            n_clusters: Nombre de clusters (auto si None)
            algorithm: Algorithme de clustering
            
        Returns:
            Résultats de la segmentation
        """
        print(f"🎯 Clustering avec {algorithm}...")
        
        if len(user_profiles) < 2:
            raise ValueError("Pas assez d'utilisateurs pour le clustering")
        
        # Détermine le nombre de clusters
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters(user_profiles)
        
        # Algorithme de clustering
        if algorithm == 'kmeans':
            clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
        elif algorithm == 'dbscan':
            # Détermine eps automatiquement
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=4)
            neighbors_fit = neighbors.fit(user_profiles)
            distances, indices = neighbors_fit.kneighbors(user_profiles)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            eps = np.percentile(distances, 90)
            
            clusterer = DBSCAN(eps=eps, min_samples=max(3, len(user_profiles) // 20))
        else:
            raise ValueError(f"Algorithme non supporté: {algorithm}")
        
        # Clustering
        cluster_labels = clusterer.fit_predict(user_profiles)
        
        # Gère les outliers de DBSCAN
        if algorithm == 'dbscan':
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            if n_clusters == 0:
                print("⚠️  DBSCAN n'a trouvé aucun cluster, utilisation de K-means")
                return self.perform_clustering(user_profiles, user_ids, 5, 'kmeans')
        
        # Calcule les métriques de qualité
        quality_metrics = {}
        
        if len(set(cluster_labels)) > 1:
            if -1 not in cluster_labels:  # Pas d'outliers
                quality_metrics['silhouette_score'] = silhouette_score(user_profiles, cluster_labels)
                quality_metrics['calinski_harabasz_score'] = calinski_harabasz_score(user_profiles, cluster_labels)
            
            if hasattr(clusterer, 'inertia_'):
                quality_metrics['inertia'] = clusterer.inertia_
        
        # Centres des clusters
        if hasattr(clusterer, 'cluster_centers_'):
            cluster_centers = clusterer.cluster_centers_
        else:
            # Calcule les centres manuellement
            cluster_centers = []
            for cluster_id in set(cluster_labels):
                if cluster_id != -1:  # Ignore les outliers
                    cluster_mask = cluster_labels == cluster_id
                    center = np.mean(user_profiles[cluster_mask], axis=0)
                    cluster_centers.append(center)
            cluster_centers = np.array(cluster_centers)
        
        # Assigne les utilisateurs aux clusters
        user_assignments = dict(zip(user_ids, cluster_labels))
        
        print(f"✅ Clustering terminé: {n_clusters} clusters identifiés")
        
        # Retourne les résultats (les segments seront créés dans analyze_segments)
        return SegmentationResults(
            segments=[],  # Sera rempli par analyze_segments
            user_assignments=user_assignments,
            cluster_centers=cluster_centers,
            quality_metrics=quality_metrics,
            feature_importance={}  # Sera calculé si possible
        )
    
    def analyze_segments(self, 
                        segmentation_results: SegmentationResults,
                        ml_data: pd.DataFrame,
                        user_queries: Dict[str, List[str]]) -> SegmentationResults:
        """
        Analyse les caractéristiques de chaque segment
        
        Args:
            segmentation_results: Résultats du clustering
            ml_data: Données ML des utilisateurs
            user_queries: Requêtes par utilisateur
            
        Returns:
            Résultats enrichis avec l'analyse des segments
        """
        print("🔬 Analyse des caractéristiques des segments...")
        
        segments = []
        cluster_labels = list(segmentation_results.user_assignments.values())
        unique_clusters = sorted(set(cluster_labels))
        
        # Supprime les outliers (-1) pour l'analyse
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        for cluster_id in unique_clusters:
            # Utilisateurs de ce cluster
            cluster_users = [user_id for user_id, label in segmentation_results.user_assignments.items() 
                           if label == cluster_id]
            
            if not cluster_users:
                continue
            
            # Données ML pour ce cluster
            cluster_ml_data = ml_data[ml_data['user_id'].isin(cluster_users)]
            
            # Caractéristiques comportementales
            characteristics = {}
            
            # Métriques moyennes
            numeric_features = [
                'search_frequency', 'avg_session_duration', 'click_through_rate',
                'conversion_rate', 'engagement_score'
            ]
            
            for feature in numeric_features:
                if feature in cluster_ml_data.columns:
                    characteristics[f'avg_{feature}'] = float(cluster_ml_data[feature].mean())
            
            # Préférences catégorielles
            category_prefs = {}
            category_features = ['prefers_electronics', 'prefers_clothing', 'prefers_home', 'prefers_sports']
            
            for feature in category_features:
                if feature in cluster_ml_data.columns:
                    category = feature.replace('prefers_', '')
                    pref_score = float(cluster_ml_data[feature].mean())
                    if pref_score > 0.3:  # Seuil de préférence
                        category_prefs[category] = pref_score
            
            characteristics['category_preferences'] = category_prefs
            
            # Sensibilité prix dominante
            price_features = ['price_sensitivity_high', 'price_sensitivity_medium', 'price_sensitivity_low']
            price_scores = {}
            
            for feature in price_features:
                if feature in cluster_ml_data.columns:
                    sensitivity = feature.replace('price_sensitivity_', '')
                    price_scores[sensitivity] = float(cluster_ml_data[feature].mean())
            
            if price_scores:
                dominant_price_sensitivity = max(price_scores, key=price_scores.get)
                characteristics['price_sensitivity'] = dominant_price_sensitivity
            
            # Requêtes représentatives
            cluster_queries = []
            for user_id in cluster_users:
                if user_id in user_queries:
                    cluster_queries.extend(user_queries[user_id])
            
            # Top requêtes du cluster
            query_counts = Counter(cluster_queries)
            representative_queries = [query for query, _ in query_counts.most_common(5)]
            
            # Score de confiance basé sur la cohésion
            if len(cluster_users) > 1:
                # Calcule la variance intra-cluster des métriques principales
                key_metrics = ['engagement_score', 'conversion_rate', 'click_through_rate']
                variances = []
                
                for metric in key_metrics:
                    if metric in cluster_ml_data.columns:
                        variance = float(cluster_ml_data[metric].var())
                        variances.append(variance)
                
                # Score de confiance inverse à la variance (plus c'est homogène, plus c'est confiant)
                avg_variance = np.mean(variances) if variances else 1.0
                confidence_score = max(0.1, min(1.0, 1.0 / (1.0 + avg_variance * 10)))
            else:
                confidence_score = 0.5  # Score neutre pour un seul utilisateur
            
            # Création du segment
            segment = UserSegment(
                segment_id=cluster_id,
                segment_name=self._generate_segment_name(characteristics, representative_queries),
                user_ids=cluster_users,
                size=len(cluster_users),
                characteristics=characteristics,
                representative_queries=representative_queries,
                confidence_score=confidence_score
            )
            
            segments.append(segment)
        
        # Met à jour les résultats
        segmentation_results.segments = segments
        
        print(f"✅ Analyse terminée: {len(segments)} segments caractérisés")
        
        # Affiche un résumé
        for segment in segments:
            print(f"  📊 {segment.segment_name}: {segment.size} utilisateurs (confiance: {segment.confidence_score:.2f})")
        
        return segmentation_results
    
    def _generate_segment_name(self, characteristics: Dict, queries: List[str]) -> str:
        """
        Génère un nom intelligent pour le segment basé sur ses caractéristiques
        """
        # Analyse des préférences catégorielles
        category_prefs = characteristics.get('category_preferences', {})
        top_category = max(category_prefs, key=category_prefs.get) if category_prefs else None
        
        # Analyse de l'engagement
        engagement = characteristics.get('avg_engagement_score', 0)
        conversion = characteristics.get('avg_conversion_rate', 0)
        
        # Analyse des requêtes pour les mots-clés
        query_text = ' '.join(queries).lower()
        
        # Logique de nommage
        if 'tech' in query_text or 'gaming' in query_text or top_category == 'electronics':
            if engagement > 0.5:
                return "Tech Enthusiasts"
            else:
                return "Tech Browsers"
        
        elif 'fashion' in query_text or 'designer' in query_text or top_category == 'clothing':
            if conversion > 0.15:
                return "Fashion Buyers"
            else:
                return "Fashion Explorers"
        
        elif 'cheap' in query_text or 'discount' in query_text or characteristics.get('price_sensitivity') == 'high':
            return "Budget Hunters"
        
        elif top_category == 'home':
            return "Home Improvers"
        
        elif top_category == 'sports':
            return "Active Lifestyle"
        
        elif engagement > 0.6 and conversion > 0.12:
            return "Premium Shoppers"
        
        elif engagement < 0.3:
            return "Casual Browsers"
        
        else:
            return "General Shoppers"
    
    def visualize_segments(self, 
                          user_profiles: np.ndarray,
                          segmentation_results: SegmentationResults,
                          user_ids: List[str],
                          save_path: Optional[str] = None) -> None:
        """
        Visualise les segments dans un espace 2D
        
        Args:
            user_profiles: Profils utilisateur
            segmentation_results: Résultats de segmentation
            user_ids: IDs des utilisateurs
            save_path: Chemin pour sauvegarder la visualisation
        """
        print("📊 Visualisation des segments...")
        
        # Réduction de dimensionnalité
        if user_profiles.shape[1] > 2:
            try:
                # Utilise UMAP si disponible, sinon t-SNE
                reducer = umap.UMAP(n_neighbors=min(15, len(user_profiles)-1), 
                                  random_state=self.random_state)
                coords_2d = reducer.fit_transform(user_profiles)
                method = "UMAP"
            except ImportError:
                try:
                    reducer = TSNE(n_components=2, random_state=self.random_state, 
                                 perplexity=min(30, len(user_profiles)-1))
                    coords_2d = reducer.fit_transform(user_profiles)
                    method = "t-SNE"
                except:
                    # Fallback vers PCA
                    reducer = PCA(n_components=2, random_state=self.random_state)
                    coords_2d = reducer.fit_transform(user_profiles)
                    method = "PCA"
        else:
            coords_2d = user_profiles
            method = "Direct"
        
        # Prépare les couleurs pour chaque segment
        cluster_labels = [segmentation_results.user_assignments[user_id] for user_id in user_ids]
        unique_labels = sorted(set(cluster_labels))
        
        # Palette de couleurs
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # Création du graphique
        plt.figure(figsize=(12, 8))
        
        for i, label in enumerate(unique_labels):
            if label == -1:  # Outliers
                color = 'black'
                marker = 'x'
                alpha = 0.5
                segment_name = "Outliers"
            else:
                color = colors[i]
                marker = 'o'
                alpha = 0.7
                # Trouve le nom du segment
                segment_name = "Unknown"
                for segment in segmentation_results.segments:
                    if segment.segment_id == label:
                        segment_name = segment.segment_name
                        break
            
            # Points du cluster
            mask = np.array(cluster_labels) == label
            plt.scatter(coords_2d[mask, 0], coords_2d[mask, 1], 
                       c=[color], label=f"{segment_name} ({np.sum(mask)})",
                       alpha=alpha, marker=marker, s=60)
        
        # Centres des clusters (si disponibles)
        if len(segmentation_results.cluster_centers) > 0 and segmentation_results.cluster_centers.shape[1] == user_profiles.shape[1]:
            try:
                if method == "UMAP":
                    centers_2d = reducer.transform(segmentation_results.cluster_centers)
                elif method == "t-SNE":
                    # t-SNE ne permet pas de transformer de nouveaux points
                    centers_2d = None
                else:
                    centers_2d = reducer.transform(segmentation_results.cluster_centers)
                
                if centers_2d is not None:
                    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                               c='red', marker='*', s=200, 
                               edgecolors='black', linewidth=2,
                               label='Centres des clusters')
            except:
                pass  # Ignore les erreurs de transformation
        
        plt.title(f'Segmentation des utilisateurs ({method})\n'
                 f'Qualité: Silhouette = {segmentation_results.quality_metrics.get("silhouette_score", "N/A"):.3f}',
                 fontsize=14, fontweight='bold')
        plt.xlabel(f'Dimension 1 ({method})')
        plt.ylabel(f'Dimension 2 ({method})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualisation sauvegardée: {save_path}")
        
        plt.show()
    
    def generate_segment_report(self, segmentation_results: SegmentationResults) -> Dict[str, Any]:
        """
        Génère un rapport détaillé des segments
        
        Args:
            segmentation_results: Résultats de la segmentation
            
        Returns:
            Rapport complet des segments
        """
        print("📋 Génération du rapport de segmentation...")
        
        if not segmentation_results.segments:
            return {"error": "Aucun segment disponible"}
        
        # Statistiques globales
        total_users = sum(segment.size for segment in segmentation_results.segments)
        
        # Rapport par segment
        segment_reports = []
        
        for segment in segmentation_results.segments:
            segment_report = {
                "segment_id": segment.segment_id,
                "segment_name": segment.segment_name,
                "size": segment.size,
                "percentage": round((segment.size / total_users) * 100, 1),
                "confidence_score": round(segment.confidence_score, 3),
                "characteristics": segment.characteristics,
                "top_search_queries": segment.representative_queries[:5],
                "insights": self._generate_segment_insights(segment)
            }
            segment_reports.append(segment_report)
        
        # Trie par taille de segment
        segment_reports.sort(key=lambda x: x["size"], reverse=True)
        
        # Métriques de qualité
        quality_summary = {
            "silhouette_score": segmentation_results.quality_metrics.get("silhouette_score"),
            "calinski_harabasz_score": segmentation_results.quality_metrics.get("calinski_harabasz_score"),
            "inertia": segmentation_results.quality_metrics.get("inertia")
        }
        
        # Comparaison entre segments
        segment_comparison = self._compare_segments(segmentation_results.segments)
        
        report = {
            "summary": {
                "total_users_segmented": total_users,
                "number_of_segments": len(segmentation_results.segments),
                "segmentation_quality": quality_summary,
                "largest_segment": segment_reports[0]["segment_name"] if segment_reports else None,
                "most_confident_segment": max(segment_reports, key=lambda x: x["confidence_score"])["segment_name"] if segment_reports else None
            },
            "segments": segment_reports,
            "segment_comparison": segment_comparison,
            "recommendations": self._generate_recommendations(segment_reports)
        }
        
        print("✅ Rapport de segmentation généré")
        return report
    
    def _generate_segment_insights(self, segment: UserSegment) -> List[str]:
        """Génère des insights pour un segment spécifique"""
        insights = []
        
        characteristics = segment.characteristics
        
        # Insights sur l'engagement
        engagement = characteristics.get('avg_engagement_score', 0)
        if engagement > 0.6:
            insights.append("Très engagés avec la plateforme")
        elif engagement < 0.3:
            insights.append("Engagement limité, potentiel d'amélioration")
        
        # Insights sur la conversion
        conversion = characteristics.get('avg_conversion_rate', 0)
        if conversion > 0.15:
            insights.append("Taux de conversion élevé")
        elif conversion < 0.05:
            insights.append("Faible taux de conversion, nécessite des incitations")
        
        # Insights sur les catégories
        category_prefs = characteristics.get('category_preferences', {})
        if category_prefs:
            top_category = max(category_prefs, key=category_prefs.get)
            insights.append(f"Forte préférence pour {top_category}")
        
        # Insights sur la sensibilité prix
        price_sensitivity = characteristics.get('price_sensitivity')
        if price_sensitivity == 'high':
            insights.append("Très sensible aux prix, privilégier les promotions")
        elif price_sensitivity == 'low':
            insights.append("Peu sensible aux prix, potentiel pour produits premium")
        
        # Insights sur les requêtes
        queries = segment.representative_queries
        if any('luxury' in q.lower() or 'premium' in q.lower() for q in queries):
            insights.append("Intéressé par les produits haut de gamme")
        
        if any('sale' in q.lower() or 'discount' in q.lower() for q in queries):
            insights.append("Recherche activement les bonnes affaires")
        
        return insights[:3]  # Limite à 3 insights principaux
    
    def _compare_segments(self, segments: List[UserSegment]) -> Dict[str, Any]:
        """Compare les segments entre eux"""
        if len(segments) < 2:
            return {}
        
        # Métriques de comparaison
        engagement_scores = []
        conversion_rates = []
        sizes = []
        confidence_scores = []
        
        for segment in segments:
            engagement_scores.append(segment.characteristics.get('avg_engagement_score', 0))
            conversion_rates.append(segment.characteristics.get('avg_conversion_rate', 0))
            sizes.append(segment.size)
            confidence_scores.append(segment.confidence_score)
        
        comparison = {
            "most_engaged_segment": segments[np.argmax(engagement_scores)].segment_name,
            "highest_converting_segment": segments[np.argmax(conversion_rates)].segment_name,
            "largest_segment": segments[np.argmax(sizes)].segment_name,
            "most_confident_segment": segments[np.argmax(confidence_scores)].segment_name,
            "engagement_variation": {
                "min": round(min(engagement_scores), 3),
                "max": round(max(engagement_scores), 3),
                "std": round(np.std(engagement_scores), 3)
            },
            "conversion_variation": {
                "min": round(min(conversion_rates), 3),
                "max": round(max(conversion_rates), 3),
                "std": round(np.std(conversion_rates), 3)
            }
        }
        
        return comparison
    
    def _generate_recommendations(self, segment_reports: List[Dict]) -> List[str]:
        """Génère des recommandations stratégiques"""
        recommendations = []
        
        if not segment_reports:
            return recommendations
        
        # Analyse des segments pour recommandations
        largest_segment = segment_reports[0]
        highest_conversion = max(segment_reports, key=lambda x: x["characteristics"].get("avg_conversion_rate", 0))
        lowest_engagement = min(segment_reports, key=lambda x: x["characteristics"].get("avg_engagement_score", 1))
        
        # Recommandations basées sur l'analyse
        recommendations.append(
            f"Concentrer les efforts marketing sur '{largest_segment['segment_name']}' "
            f"qui représente {largest_segment['percentage']}% des utilisateurs"
        )
        
        if highest_conversion["characteristics"].get("avg_conversion_rate", 0) > 0.15:
            recommendations.append(
                f"Développer des produits premium pour '{highest_conversion['segment_name']}' "
                f"avec un taux de conversion de {highest_conversion['characteristics'].get('avg_conversion_rate', 0):.1%}"
            )
        
        if lowest_engagement["characteristics"].get("avg_engagement_score", 1) < 0.3:
            recommendations.append(
                f"Améliorer l'engagement du segment '{lowest_engagement['segment_name']}' "
                f"avec des campagnes de re-engagement ciblées"
            )
        
        # Recommandations générales
        price_sensitive_segments = [s for s in segment_reports 
                                  if s["characteristics"].get("price_sensitivity") == "high"]
        if price_sensitive_segments:
            recommendations.append(
                "Mettre en place des programmes de fidélité et promotions "
                f"pour les {len(price_sensitive_segments)} segments sensibles aux prix"
            )
        
        return recommendations
    
    def export_segmentation_results(self, 
                                   segmentation_results: SegmentationResults,
                                   output_dir: str = ".") -> Dict[str, str]:
        """
        Exporte les résultats de segmentation
        
        Args:
            segmentation_results: Résultats de la segmentation
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire avec les chemins des fichiers exportés
        """
        import os
        
        print(f"💾 Export des résultats de segmentation vers {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        try:
            # Export des assignations utilisateur-segment
            assignments_data = []
            for user_id, segment_id in segmentation_results.user_assignments.items():
                # Trouve le nom du segment
                segment_name = "Unknown"
                for segment in segmentation_results.segments:
                    if segment.segment_id == segment_id:
                        segment_name = segment.segment_name
                        break
                
                assignments_data.append({
                    'user_id': user_id,
                    'segment_id': segment_id,
                    'segment_name': segment_name
                })
            
            assignments_file = os.path.join(output_dir, "user_segment_assignments.csv")
            pd.DataFrame(assignments_data).to_csv(assignments_file, index=False)
            exported_files["assignments"] = assignments_file
            
            # Export des caractéristiques des segments
            segments_data = []
            for segment in segmentation_results.segments:
                segment_dict = {
                    'segment_id': segment.segment_id,
                    'segment_name': segment.segment_name,
                    'size': segment.size,
                    'confidence_score': segment.confidence_score,
                    'representative_queries': '; '.join(segment.representative_queries),
                }
            for key, value in segment.characteristics.items():
                if isinstance(value, dict):
                    segment_dict[key] = json.dumps(value)
                elif isinstance(value, (np.integer, np.floating, np.ndarray)):
                    segment_dict[key] = float(value)
                else:
                    segment_dict[key] = value
                
                segments_data.append(segment_dict)
            
            segments_file = os.path.join(output_dir, "segment_characteristics.csv")
            pd.DataFrame(segments_data).to_csv(segments_file, index=False)
            exported_files["segments"] = segments_file
            
            # Export du rapport complet
            report = self.generate_segment_report(segmentation_results)
            report_file = os.path.join(output_dir, "segmentation_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            exported_files["report"] = report_file
            
            print(f"✅ Résultats exportés: {len(exported_files)} fichiers")
            return exported_files
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'export: {e}")
            return {}

def main():
    """
    Fonction principale pour tester la segmentation IA
    """
    print("🚀 MOTEUR DE SEGMENTATION IA")
    print("=" * 50)
    
    # Initialisation
    ai_engine = AISegmentationEngine(embedding_model=EMBEDDING_MODEL)
    
    # Test avec des données simulées si pas de données réelles
    try:
        # Charge les données depuis search_extractor si disponible
        from search_extractor import SearchDataExtractor
        
        extractor = SearchDataExtractor()
        search_data = extractor.extract_user_search_history(days_back=60)
        user_metrics = extractor.aggregate_user_behavior_patterns(search_data)
        ml_data = extractor.prepare_data_for_ml(user_metrics)
        user_queries = extractor.get_search_queries_for_embeddings(search_data)
        
        print(f"📊 Données chargées: {len(ml_data)} utilisateurs")
        
    except Exception as e:
        print(f"⚠️  Erreur de chargement des données: {e}")
        print("🎭 Utilisation de données de test...")
        return
    
    if ml_data.empty or not user_queries:
        print("❌ Pas de données disponibles pour la segmentation")
        return
    
    # 1. Génération des embeddings
    print("\n🔤 GÉNÉRATION DES EMBEDDINGS")
    user_embeddings = ai_engine.generate_search_embeddings(user_queries)
    
    # 2. Création des profils utilisateur
    print("\n👤 CRÉATION DES PROFILS UTILISATEUR")
    user_profiles, user_ids = ai_engine.create_user_profiles(ml_data, user_embeddings)
    
    # 3. Clustering
    print("\n🎯 CLUSTERING DES UTILISATEURS")
    segmentation_results = ai_engine.perform_clustering(user_profiles, user_ids)
    
    # 4. Analyse des segments
    print("\n🔬 ANALYSE DES SEGMENTS")
    segmentation_results = ai_engine.analyze_segments(segmentation_results, ml_data, user_queries)
    
    # 5. Visualisation
    print("\n📊 VISUALISATION DES SEGMENTS")
    try:
        ai_engine.visualize_segments(user_profiles, segmentation_results, user_ids, 
                                   save_path="user_segments_visualization.png")
    except Exception as e:
        print(f"⚠️  Erreur de visualisation: {e}")
    
    # 6. Génération du rapport
    print("\n📋 GÉNÉRATION DU RAPPORT")
    report = ai_engine.generate_segment_report(segmentation_results)
    
    # Affichage des résultats
    print(f"\n🎯 RÉSULTATS DE LA SEGMENTATION:")
    print(f"  • {report['summary']['number_of_segments']} segments identifiés")
    print(f"  • {report['summary']['total_users_segmented']} utilisateurs segmentés")
    
    if 'silhouette_score' in report['summary']['segmentation_quality']:
        print(f"  • Qualité (Silhouette): {report['summary']['segmentation_quality']['silhouette_score']:.3f}")
    
    print(f"\n📊 SEGMENTS IDENTIFIÉS:")
    for segment_info in report['segments']:
        print(f"  • {segment_info['segment_name']}: {segment_info['size']} utilisateurs ({segment_info['percentage']}%)")
        print(f"    Confiance: {segment_info['confidence_score']:.2f}")
        if segment_info['insights']:
            print(f"    Insights: {', '.join(segment_info['insights'])}")
        print()
    
    print(f"💡 RECOMMANDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # 7. Export des résultats
    print(f"\n💾 EXPORT DES RÉSULTATS")
    exported_files = ai_engine.export_segmentation_results(segmentation_results)
    
    if exported_files:
        print(f"Fichiers exportés:")
        for file_type, file_path in exported_files.items():
            print(f"  • {file_type}: {file_path}")
    
    print(f"\n✅ Segmentation IA terminée!")
    print(f"🎯 Prochaine étape: Indexation dans Elasticsearch avec elasticsearch_indexer.py")
    
    return segmentation_results, report

if __name__ == "__main__":
    main()