import json
import pandas as pd
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, Counter
import re
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserBehaviorMetrics:
    """Structure pour les métriques comportementales d'un utilisateur"""
    user_id: str
    search_frequency: float
    avg_session_duration: float
    preferred_categories: List[str]
    search_query_complexity: float
    click_through_rate: float
    conversion_rate: float
    brand_loyalty_score: float
    price_sensitivity: str
    peak_activity_hours: List[int]

class SearchDataExtractor:
    """
    Extracteur de données de recherche depuis Elasticsearch
    Prépare les données pour l'analyse comportementale et la segmentation IA
    """
    
    def __init__(self, es_host='localhost', es_port=9200, index_name='user_sessions'):
        """
        Initialise l'extracteur avec connexion Elasticsearch
        
        Args:
            es_host: Host Elasticsearch
            es_port: Port Elasticsearch  
            index_name: Nom de l'index des sessions utilisateur
        """
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = index_name
        self.client = None
        self.connect()
    
    def connect(self):
        """Établit la connexion à Elasticsearch"""
        try:
            self.client = Elasticsearch(
                f"http://{self.es_host}:{self.es_port}",
                request_timeout=30,
                retry_on_timeout=True
            )
            
            if self.client.ping():
                logger.info("✅ Connexion Elasticsearch établie")
                return True
            else:
                logger.error("❌ Impossible de se connecter à Elasticsearch")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur de connexion Elasticsearch: {e}")
            return False
    
    def create_user_sessions_index(self):
        """
        Crée l'index pour les sessions utilisateur si il n'existe pas
        """
        print(f"🔧 Création de l'index: {self.index_name}")
        
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "search_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "session_id": {"type": "keyword"},
                    "search_query": {
                        "type": "text",
                        "analyzer": "search_analyzer",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "clicked_product_ids": {"type": "keyword"},
                    "viewed_categories": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "session_duration": {"type": "integer"},
                    "conversion_event": {"type": "boolean"},
                    "purchase_amount": {"type": "double"},
                    "device_type": {"type": "keyword"},
                    "search_filters_used": {"type": "nested"},
                    "page_views": {"type": "integer"},
                    "time_on_page": {"type": "integer"}
                }
            }
        }
        
        try:
            if self.client.indices.exists(index=self.index_name):
                print(f"ℹ️  Index {self.index_name} existe déjà")
                return True
            
            response = self.client.indices.create(
                index=self.index_name,
                body=mapping
            )
            
            print(f"✅ Index créé avec succès: {self.index_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'index: {e}")
            return False
    
    def extract_user_search_history(self, 
                                  user_ids: Optional[List[str]] = None, 
                                  days_back: int = 30,
                                  max_results: int = 10000) -> pd.DataFrame:
        """
        Extrait l'historique de recherche des utilisateurs
        
        Args:
            user_ids: Liste des IDs utilisateur (None pour tous)
            days_back: Nombre de jours à récupérer
            max_results: Nombre maximum de résultats
            
        Returns:
            DataFrame avec l'historique de recherche
        """
        print(f"📊 Extraction de l'historique de recherche ({days_back} derniers jours)")
        
        # Construction de la requête Elasticsearch
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": start_date.strftime('%Y-%m-%d'),
                                    "lte": end_date.strftime('%Y-%m-%d')
                                }
                            }
                        }
                    ]
                }
            },
            "sort": [{"timestamp": {"order": "desc"}}],
            "size": max_results
        }
        
        # Filtre par utilisateurs si spécifié
        if user_ids:
            query["query"]["bool"]["must"].append({
                "terms": {"user_id": user_ids}
            })
        
        try:
            # Recherche avec scroll pour traiter de gros volumes
            response = self.client.search(
                index=self.index_name,
                body=query,
                scroll='2m'
            )
            
            all_hits = []
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            
            while hits:
                all_hits.extend([hit['_source'] for hit in hits])
                
                # Continue le scroll
                response = self.client.scroll(
                    scroll_id=scroll_id,
                    scroll='2m'
                )
                hits = response['hits']['hits']
            
            # Nettoyage du scroll
            self.client.clear_scroll(scroll_id=scroll_id)
            
            if not all_hits:
                print("⚠️  Aucune donnée trouvée. Génération de données de test...")
                return self._generate_sample_data(days_back)
            
            # Conversion en DataFrame
            df = pd.DataFrame(all_hits)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            print(f"✅ {len(df)} sessions extraites")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction: {e}")
            print("⚠️  Génération de données de test à la place...")
            return self._generate_sample_data(days_back)
    
    def _generate_sample_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Génère des données de test réalistes pour la démonstration
        """
        print("🎭 Génération de données de test réalistes...")
        
        # Personas utilisateur
        user_personas = {
            'tech_enthusiast': {
                'queries': ['wireless headphones', 'gaming laptop', 'smartphone', 'tablet', 'smartwatch', 'bluetooth speaker'],
                'categories': ['electronics', 'gaming', 'tech'],
                'behavior': {'click_rate': 0.8, 'conversion_rate': 0.15, 'session_duration': 180}
            },
            'fashion_forward': {
                'queries': ['designer dress', 'summer fashion', 'luxury handbag', 'trendy shoes', 'jewelry'],
                'categories': ['clothing', 'fashion', 'accessories'],
                'behavior': {'click_rate': 0.7, 'conversion_rate': 0.12, 'session_duration': 240}
            },
            'budget_conscious': {
                'queries': ['cheap laptop', 'discount', 'sale', 'best price', 'affordable'],
                'categories': ['electronics', 'home', 'clothing'],
                'behavior': {'click_rate': 0.6, 'conversion_rate': 0.08, 'session_duration': 120}
            },
            'home_decorator': {
                'queries': ['furniture', 'home decor', 'kitchen appliances', 'bedding', 'lighting'],
                'categories': ['home', 'furniture', 'decor'],
                'behavior': {'click_rate': 0.65, 'conversion_rate': 0.10, 'session_duration': 200}
            },
            'sports_fanatic': {
                'queries': ['running shoes', 'fitness equipment', 'yoga mat', 'protein powder'],
                'categories': ['sports', 'fitness', 'health'],
                'behavior': {'click_rate': 0.75, 'conversion_rate': 0.13, 'session_duration': 150}
            }
        }
        
        sample_data = []
        end_date = datetime.now()
        
        # Génère des utilisateurs et leurs sessions
        for user_id in range(1, 501):  # 500 utilisateurs
            # Assigne un persona aléatoire
            persona_name = np.random.choice(list(user_personas.keys()))
            persona = user_personas[persona_name]
            
            # Génère des sessions pour cet utilisateur
            num_sessions = np.random.poisson(days_back // 5)  # Moyenne d'une session tous les 5 jours
            
            for session in range(num_sessions):
                # Date aléatoire dans la période
                days_ago = np.random.randint(0, days_back)
                timestamp = end_date - timedelta(days=days_ago, 
                                               hours=np.random.randint(0, 24),
                                               minutes=np.random.randint(0, 60))
                
                # Requête de recherche basée sur le persona
                search_query = np.random.choice(persona['queries'])
                
                # Produits cliqués (simulation)
                num_clicks = np.random.poisson(2) if np.random.random() < persona['behavior']['click_rate'] else 0
                clicked_products = [f"p_{np.random.randint(1, 1000)}" for _ in range(num_clicks)]
                
                # Catégories vues
                viewed_categories = np.random.choice(persona['categories'], 
                                                   size=min(len(persona['categories']), np.random.randint(1, 3)),
                                                   replace=False).tolist()
                
                # Conversion
                conversion = np.random.random() < persona['behavior']['conversion_rate']
                purchase_amount = np.random.normal(100, 50) if conversion else 0
                
                session_data = {
                    'user_id': f"user_{user_id:04d}",
                    'session_id': f"session_{user_id}_{session}",
                    'search_query': search_query,
                    'clicked_product_ids': clicked_products,
                    'viewed_categories': viewed_categories,
                    'timestamp': timestamp,
                    'session_duration': int(np.random.normal(persona['behavior']['session_duration'], 60)),
                    'conversion_event': conversion,
                    'purchase_amount': max(0, purchase_amount),
                    'device_type': np.random.choice(['desktop', 'mobile', 'tablet'], p=[0.4, 0.5, 0.1]),
                    'page_views': np.random.poisson(5),
                    'time_on_page': np.random.randint(30, 300),
                    'persona': persona_name  # Pour validation
                }
                
                sample_data.append(session_data)
        
        df = pd.DataFrame(sample_data)
        print(f"✅ {len(df)} sessions de test générées pour {df['user_id'].nunique()} utilisateurs")
        
        return df
    
    def aggregate_user_behavior_patterns(self, search_data: pd.DataFrame) -> List[UserBehaviorMetrics]:
        """
        Agrège les patterns comportementaux par utilisateur
        
        Args:
            search_data: DataFrame avec les données de recherche
            
        Returns:
            Liste des métriques comportementales par utilisateur
        """
        print("🔍 Analyse des patterns comportementaux...")
        
        user_metrics = []
        
        for user_id in search_data['user_id'].unique():
            user_data = search_data[search_data['user_id'] == user_id]
            
            # Métriques de base
            total_sessions = len(user_data)
            total_days = (user_data['timestamp'].max() - user_data['timestamp'].min()).days + 1
            search_frequency = total_sessions / max(total_days, 1)
            
            # Durée moyenne de session
            avg_session_duration = user_data['session_duration'].mean()
            
            # Catégories préférées
            all_categories = []
            for categories in user_data['viewed_categories'].dropna():
                if isinstance(categories, list):
                    all_categories.extend(categories)
                else:
                    all_categories.append(categories)
            
            category_counts = Counter(all_categories)
            preferred_categories = [cat for cat, _ in category_counts.most_common(3)]
            
            # Complexité des requêtes (nombre de mots moyen)
            query_complexity = user_data['search_query'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            ).mean()
            
            # Taux de clic (sessions avec clicks / total sessions)
            sessions_with_clicks = user_data[user_data['clicked_product_ids'].apply(
                lambda x: len(x) > 0 if isinstance(x, list) else False
            )]
            click_through_rate = len(sessions_with_clicks) / total_sessions
            
            # Taux de conversion
            conversions = user_data[user_data['conversion_event'] == True]
            conversion_rate = len(conversions) / total_sessions
            
            # Score de fidélité marque (basé sur la répétition des recherches)
            unique_queries = user_data['search_query'].nunique()
            brand_loyalty_score = 1 - (unique_queries / total_sessions)
            
            # Sensibilité prix (basée sur les mots-clés)
            price_keywords = ['cheap', 'discount', 'sale', 'affordable', 'budget']
            price_sensitive_searches = user_data[user_data['search_query'].str.contains(
                '|'.join(price_keywords), case=False, na=False
            )]
            price_sensitivity = "high" if len(price_sensitive_searches) / total_sessions > 0.3 else \
                              "medium" if len(price_sensitive_searches) / total_sessions > 0.1 else "low"
            
            # Heures d'activité peak
            user_data['hour'] = user_data['timestamp'].dt.hour
            hour_counts = user_data['hour'].value_counts()
            peak_activity_hours = hour_counts.head(3).index.tolist()
            
            metrics = UserBehaviorMetrics(
                user_id=user_id,
                search_frequency=round(search_frequency, 2),
                avg_session_duration=round(avg_session_duration, 1),
                preferred_categories=preferred_categories,
                search_query_complexity=round(query_complexity, 2),
                click_through_rate=round(click_through_rate, 3),
                conversion_rate=round(conversion_rate, 3),
                brand_loyalty_score=round(brand_loyalty_score, 3),
                price_sensitivity=price_sensitivity,
                peak_activity_hours=peak_activity_hours
            )
            
            user_metrics.append(metrics)
        
        print(f"✅ Patterns analysés pour {len(user_metrics)} utilisateurs")
        return user_metrics
    
    def prepare_data_for_ml(self, user_metrics: List[UserBehaviorMetrics]) -> pd.DataFrame:
        """
        Prépare les données pour le machine learning
        
        Args:
            user_metrics: Liste des métriques utilisateur
            
        Returns:
            DataFrame formaté pour ML
        """
        print("🤖 Préparation des données pour le ML...")
        
        # Conversion en DataFrame
        ml_data = []
        
        for metrics in user_metrics:
            # Features numériques
            features = {
                'user_id': metrics.user_id,
                'search_frequency': metrics.search_frequency,
                'avg_session_duration': metrics.avg_session_duration,
                'search_query_complexity': metrics.search_query_complexity,
                'click_through_rate': metrics.click_through_rate,
                'conversion_rate': metrics.conversion_rate,
                'brand_loyalty_score': metrics.brand_loyalty_score,
                'num_preferred_categories': len(metrics.preferred_categories),
                'num_peak_hours': len(metrics.peak_activity_hours),
            }
            
            # Encoding des catégories préférées (one-hot pour les principales)
            main_categories = ['electronics', 'clothing', 'home', 'sports', 'books']
            for cat in main_categories:
                features[f'prefers_{cat}'] = 1 if cat in metrics.preferred_categories else 0
            
            # Encoding de la sensibilité prix
            features['price_sensitivity_high'] = 1 if metrics.price_sensitivity == 'high' else 0
            features['price_sensitivity_medium'] = 1 if metrics.price_sensitivity == 'medium' else 0
            features['price_sensitivity_low'] = 1 if metrics.price_sensitivity == 'low' else 0
            
            # Features dérivées
            features['engagement_score'] = (
                metrics.click_through_rate * 0.4 +
                metrics.conversion_rate * 0.6
            )
            
            features['activity_intensity'] = (
                metrics.search_frequency * metrics.avg_session_duration / 100
            )
            
            ml_data.append(features)
        
        df = pd.DataFrame(ml_data)
        
        # Normalisation des features numériques
        numeric_features = [
            'search_frequency', 'avg_session_duration', 'search_query_complexity',
            'click_through_rate', 'conversion_rate', 'brand_loyalty_score',
            'engagement_score', 'activity_intensity'
        ]
        
        for feature in numeric_features:
            if feature in df.columns:
                # Min-max scaling
                min_val = df[feature].min()
                max_val = df[feature].max()
                if max_val > min_val:
                    df[f'{feature}_normalized'] = (df[feature] - min_val) / (max_val - min_val)
                else:
                    df[f'{feature}_normalized'] = 0
        
        # Gestion des valeurs manquantes
        df = df.fillna(0)
        
        print(f"✅ Données ML préparées: {len(df)} utilisateurs, {len(df.columns)} features")
        return df
    
    def get_search_queries_for_embeddings(self, search_data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Extrait les requêtes de recherche par utilisateur pour les embeddings
        
        Args:
            search_data: DataFrame avec les données de recherche
            
        Returns:
            Dictionnaire {user_id: [liste des requêtes]}
        """
        print("📝 Extraction des requêtes pour embeddings...")
        
        user_queries = {}
        
        for user_id in search_data['user_id'].unique():
            user_data = search_data[search_data['user_id'] == user_id]
            
            # Récupère toutes les requêtes uniques de l'utilisateur
            queries = user_data['search_query'].dropna().unique().tolist()
            
            # Nettoie et filtre les requêtes
            cleaned_queries = []
            for query in queries:
                if isinstance(query, str) and len(query.strip()) > 2:
                    cleaned_queries.append(query.strip().lower())
            
            if cleaned_queries:
                user_queries[user_id] = cleaned_queries
        
        total_queries = sum(len(queries) for queries in user_queries.values())
        print(f"✅ {total_queries} requêtes extraites pour {len(user_queries)} utilisateurs")
        
        return user_queries
    
    def generate_user_behavior_report(self, user_metrics: List[UserBehaviorMetrics]) -> Dict[str, Any]:
        """
        Génère un rapport d'analyse comportementale
        
        Args:
            user_metrics: Liste des métriques utilisateur
            
        Returns:
            Dictionnaire avec le rapport complet
        """
        print("📊 Génération du rapport comportemental...")
        
        if not user_metrics:
            return {"error": "Aucune donnée disponible"}
        
        # Métriques globales
        search_frequencies = [m.search_frequency for m in user_metrics]
        conversion_rates = [m.conversion_rate for m in user_metrics]
        session_durations = [m.avg_session_duration for m in user_metrics]
        
        # Analyse des catégories
        all_categories = []
        for m in user_metrics:
            all_categories.extend(m.preferred_categories)
        
        category_counts = Counter(all_categories)
        
        # Analyse de la sensibilité prix
        price_sensitivity_dist = Counter([m.price_sensitivity for m in user_metrics])
        
        report = {
            "summary": {
                "total_users": len(user_metrics),
                "avg_search_frequency": round(np.mean(search_frequencies), 2),
                "avg_conversion_rate": round(np.mean(conversion_rates), 3),
                "avg_session_duration": round(np.mean(session_durations), 1)
            },
            "distributions": {
                "search_frequency": {
                    "min": round(min(search_frequencies), 2),
                    "max": round(max(search_frequencies), 2),
                    "median": round(np.median(search_frequencies), 2),
                    "std": round(np.std(search_frequencies), 2)
                },
                "conversion_rates": {
                    "min": round(min(conversion_rates), 3),
                    "max": round(max(conversion_rates), 3),
                    "median": round(np.median(conversion_rates), 3),
                    "std": round(np.std(conversion_rates), 3)
                }
            },
            "top_categories": [
                {"category": cat, "users": count}
                for cat, count in category_counts.most_common(10)
            ],
            "price_sensitivity": dict(price_sensitivity_dist),
            "engagement_segments": {
                "high_engagement": len([m for m in user_metrics 
                                      if m.click_through_rate > 0.5 and m.conversion_rate > 0.1]),
                "medium_engagement": len([m for m in user_metrics 
                                        if 0.2 < m.click_through_rate <= 0.5 or 0.05 < m.conversion_rate <= 0.1]),
                "low_engagement": len([m for m in user_metrics 
                                     if m.click_through_rate <= 0.2 and m.conversion_rate <= 0.05])
            }
        }
        
        print("✅ Rapport comportemental généré")
        return report
    
    def export_data(self, 
                   search_data: pd.DataFrame, 
                   user_metrics: List[UserBehaviorMetrics], 
                   ml_data: pd.DataFrame,
                   output_dir: str = ".") -> Dict[str, str]:
        """
        Exporte les données analysées
        
        Args:
            search_data: Données brutes de recherche
            user_metrics: Métriques comportementales
            ml_data: Données préparées pour ML
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire avec les chemins des fichiers exportés
        """
        import os
        
        print(f"💾 Export des données vers {output_dir}...")
        
        # Assure que le répertoire existe
        os.makedirs(output_dir, exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export des données brutes
            raw_file = os.path.join(output_dir, "search_data_raw.csv")
            search_data.to_csv(raw_file, index=False)
            exported_files["raw_data"] = raw_file
            
            # Export des métriques utilisateur
            metrics_data = []
            for metrics in user_metrics:
                metrics_dict = {
                    'user_id': metrics.user_id,
                    'search_frequency': metrics.search_frequency,
                    'avg_session_duration': metrics.avg_session_duration,
                    'preferred_categories': ','.join(metrics.preferred_categories),
                    'search_query_complexity': metrics.search_query_complexity,
                    'click_through_rate': metrics.click_through_rate,
                    'conversion_rate': metrics.conversion_rate,
                    'brand_loyalty_score': metrics.brand_loyalty_score,
                    'price_sensitivity': metrics.price_sensitivity,
                    'peak_activity_hours': ','.join(map(str, metrics.peak_activity_hours))
                }
                metrics_data.append(metrics_dict)
            
            metrics_file = os.path.join(output_dir, "user_behavior_metrics.csv")
            pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
            exported_files["metrics"] = metrics_file
            
            # Export des données ML
            ml_file = os.path.join(output_dir, "ml_features.csv")
            ml_data.to_csv(ml_file, index=False)
            exported_files["ml_data"] = ml_file
            
            print(f"✅ Données exportées: {len(exported_files)} fichiers")
            return exported_files
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'export: {e}")
            return {}

def main():
    """
    Fonction principale pour tester l'extraction de données
    """
    print("🚀 EXTRACTEUR DE DONNÉES DE RECHERCHE")
    print("=" * 50)
    
    # Initialisation
    extractor = SearchDataExtractor()
    
    # Test de connexion
    if not extractor.client or not extractor.client.ping():
        print("❌ Elasticsearch non disponible - utilisation de données de test")
    
    # Création de l'index (si nécessaire)
    extractor.create_user_sessions_index()
    
    # Extraction des données
    search_data = extractor.extract_user_search_history(days_back=60)
    
    if search_data.empty:
        print("❌ Aucune donnée extraite")
        return
    
    print(f"\n📊 Aperçu des données:")
    print(f"  • {len(search_data)} sessions")
    print(f"  • {search_data['user_id'].nunique()} utilisateurs uniques")
    print(f"  • Période: {search_data['timestamp'].min()} à {search_data['timestamp'].max()}")
    
    # Analyse comportementale
    user_metrics = extractor.aggregate_user_behavior_patterns(search_data)
    
    # Préparation pour ML
    ml_data = extractor.prepare_data_for_ml(user_metrics)
    
    # Extraction des requêtes pour embeddings
    user_queries = extractor.get_search_queries_for_embeddings(search_data)
    
    # Génération du rapport
    report = extractor.generate_user_behavior_report(user_metrics)
    
    print(f"\n📈 RAPPORT COMPORTEMENTAL:")
    print(f"  • Utilisateurs analysés: {report['summary']['total_users']}")
    print(f"  • Fréquence de recherche moyenne: {report['summary']['avg_search_frequency']}")
    print(f"  • Taux de conversion moyen: {report['summary']['avg_conversion_rate']:.1%}")
    print(f"  • Durée de session moyenne: {report['summary']['avg_session_duration']:.1f}s")
    
    print(f"\n🏷️  TOP CATÉGORIES:")
    for cat_info in report['top_categories'][:5]:
        print(f"  • {cat_info['category']}: {cat_info['users']} utilisateurs")
    
    print(f"\n💰 SENSIBILITÉ PRIX:")
    for sensitivity, count in report['price_sensitivity'].items():
        print(f"  • {sensitivity}: {count} utilisateurs")
    
    # Export des données
    exported_files = extractor.export_data(search_data, user_metrics, ml_data)
    
    if exported_files:
        print(f"\n💾 FICHIERS EXPORTÉS:")
        for file_type, file_path in exported_files.items():
            print(f"  • {file_type}: {file_path}")
    
    print(f"\n✅ Extraction et analyse terminées!")
    print(f"🎯 Prochaine étape: Segmentation IA avec ai_segmentation.py")
    
    return {
        'search_data': search_data,
        'user_metrics': user_metrics,
        'ml_data': ml_data,
        'user_queries': user_queries,
        'report': report
    }

if __name__ == "__main__":
    main()