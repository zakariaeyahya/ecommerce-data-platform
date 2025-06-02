from elasticsearch import Elasticsearch
import json
import pandas as pd
from datetime import datetime
import logging

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationIndexer:
    """
    Indexeur simple pour stocker les segments utilisateur dans Elasticsearch
    """
    
    def __init__(self, es_host='localhost', es_port=9200):
        """Initialise la connexion Elasticsearch"""
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = 'user_segments'
        self.client = None
        self.connect()
    
    def connect(self):
        """Connexion à Elasticsearch"""
        try:
            self.client = Elasticsearch(f"http://{self.es_host}:{self.es_port}")
            if self.client.ping():
                print("✅ Connexion Elasticsearch OK")
                return True
            else:
                print("❌ Connexion Elasticsearch échouée")
                return False
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")
            return False
    
    def create_segments_index(self):
        """Crée l'index pour les segments"""
        print(f"🔧 Création de l'index: {self.index_name}")
        
        # Mapping simple
        mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "segment_id": {"type": "integer"},
                    "segment_name": {"type": "keyword"},
                    "confidence_score": {"type": "float"},
                    "created_at": {"type": "date"},
                    "segment_features": {
                        "properties": {
                            "avg_engagement": {"type": "float"},
                            "avg_conversion": {"type": "float"},
                            "preferred_category": {"type": "keyword"},
                            "price_sensitivity": {"type": "keyword"}
                        }
                    }
                }
            }
        }
        
        try:
            # Supprime l'index s'il existe
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                print(f"🗑️  Index existant supprimé")
            
            # Crée le nouvel index
            self.client.indices.create(index=self.index_name, body=mapping)
            print(f"✅ Index créé: {self.index_name}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur création index: {e}")
            return False
    
    def index_user_segments(self, assignments_file="user_segment_assignments.csv"):
        """
        Indexe les segments utilisateur depuis le fichier CSV
        
        Args:
            assignments_file: Fichier CSV avec les assignations utilisateur-segment
        """
        print(f"📦 Indexation des segments depuis {assignments_file}")
        
        try:
            # Charge les données
            df = pd.read_csv(assignments_file)
            print(f"📊 {len(df)} utilisateurs à indexer")
            
            # Prépare les documents pour l'indexation
            docs = []
            for _, row in df.iterrows():
                doc = {
                    "user_id": row['user_id'],
                    "segment_id": int(row['segment_id']),
                    "segment_name": row['segment_name'],
                    "confidence_score": 0.85,  # Score par défaut
                    "created_at": datetime.now().isoformat(),
                    "segment_features": {
                        "avg_engagement": 0.5,  # Valeurs par défaut
                        "avg_conversion": 0.1,
                        "preferred_category": "general",
                        "price_sensitivity": "medium"
                    }
                }
                docs.append(doc)
            
            # Indexation en lot
            success_count = 0
            for doc in docs:
                try:
                    response = self.client.index(
                        index=self.index_name,
                        id=doc['user_id'],
                        body=doc
                    )
                    success_count += 1
                except Exception as e:
                    print(f"❌ Erreur indexation {doc['user_id']}: {e}")
            
            print(f"✅ {success_count}/{len(docs)} utilisateurs indexés")
            return success_count
            
        except Exception as e:
            print(f"❌ Erreur lors de l'indexation: {e}")
            return 0
    
    def search_users_by_segment(self, segment_name, size=10):
        """
        Recherche des utilisateurs par segment
        
        Args:
            segment_name: Nom du segment à rechercher
            size: Nombre de résultats
        """
        print(f"🔍 Recherche utilisateurs du segment: {segment_name}")
        
        query = {
            "query": {
                "term": {"segment_name": segment_name}
            },
            "size": size
        }
        
        try:
            response = self.client.search(index=self.index_name, body=query)
            hits = response['hits']['hits']
            
            users = []
            for hit in hits:
                users.append({
                    'user_id': hit['_source']['user_id'],
                    'segment_name': hit['_source']['segment_name'],
                    'confidence': hit['_source']['confidence_score']
                })
            
            print(f"✅ {len(users)} utilisateurs trouvés")
            return users
            
        except Exception as e:
            print(f"❌ Erreur recherche: {e}")
            return []
    
    def get_segment_stats(self):
        """Statistiques simples des segments"""
        print("📊 Calcul des statistiques des segments")
        
        query = {
            "size": 0,
            "aggs": {
                "segments": {
                    "terms": {"field": "segment_name"}
                }
            }
        }
        
        try:
            response = self.client.search(index=self.index_name, body=query)
            buckets = response['aggregations']['segments']['buckets']
            
            stats = {}
            for bucket in buckets:
                stats[bucket['key']] = bucket['doc_count']
            
            print("📊 Répartition des segments:")
            for segment, count in stats.items():
                print(f"  • {segment}: {count} utilisateurs")
            
            return stats
            
        except Exception as e:
            print(f"❌ Erreur statistiques: {e}")
            return {}
    
    def test_search_functionality(self):
        """Test rapide des fonctionnalités de recherche"""
        print("\n🧪 TEST DES FONCTIONNALITÉS")
        print("=" * 30)
        
        # 1. Statistiques globales
        stats = self.get_segment_stats()
        
        # 2. Recherche par segment
        if stats:
            first_segment = list(stats.keys())[0]
            users = self.search_users_by_segment(first_segment, 3)
            
            if users:
                print(f"\n🔍 Exemples d'utilisateurs du segment '{first_segment}':")
                for user in users[:3]:
                    print(f"  • {user['user_id']} (confiance: {user['confidence']})")
        
        # 3. Recherche générale
        try:
            response = self.client.search(
                index=self.index_name,
                body={"query": {"match_all": {}}, "size": 5}
            )
            total = response['hits']['total']['value']
            print(f"\n📊 Total utilisateurs indexés: {total}")
            
        except Exception as e:
            print(f"❌ Erreur test: {e}")

def main():
    """Fonction principale"""
    print("🚀 INDEXEUR ELASTICSEARCH - SEGMENTS UTILISATEUR")
    print("=" * 50)
    
    # Initialisation
    indexer = SegmentationIndexer()
    
    if not indexer.client or not indexer.client.ping():
        print("❌ Elasticsearch non disponible")
        return
    
    # 1. Création de l'index
    if not indexer.create_segments_index():
        print("❌ Impossible de créer l'index")
        return
    
    # 2. Indexation des segments
    success_count = indexer.index_user_segments()
    
    if success_count == 0:
        print("❌ Aucun segment indexé")
        return
    
    # 3. Test des fonctionnalités
    indexer.test_search_functionality()
    
    print("\n✅ Indexation terminée avec succès!")
    print("\n🎯 Les segments sont maintenant disponibles dans Elasticsearch")
    print("💡 Utilisable avec l'API Flask pour des requêtes en temps réel")

if __name__ == "__main__":
    main()