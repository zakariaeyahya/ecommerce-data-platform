from elasticsearch import Elasticsearch
from datetime import datetime
import json
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """
    Client Elasticsearch pour la synchronisation des commandes
    """
    
    def __init__(self, host='localhost', port=9200):
        """
        Initialise la connexion Elasticsearch
        """
        self.host = host
        self.port = port
        self.client = None
        self.connect()
    
    def connect(self):
        """
        Établit la connexion à Elasticsearch
        """
        try:
            # Configuration pour les versions récentes d'Elasticsearch
            self.client = Elasticsearch(
                f"http://{self.host}:{self.port}",
                request_timeout=30,
                retry_on_timeout=True
            )
            
            # Test de connexion
            if self.client.ping():
                logger.info("✅ Connexion Elasticsearch établie")
                return True
            else:
                logger.error("❌ Impossible de se connecter à Elasticsearch")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur de connexion Elasticsearch: {e}")
            return False
    
    def create_orders_index(self, index_name='ecommerce_orders'):
        """
        Crée l'index pour les commandes avec mapping optimisé
        """
        print(f"🔧 Création de l'index: {index_name}")
        
        # Mapping pour l'index des commandes
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ecommerce_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "order_id": {
                        "type": "long"
                    },
                    "store_id": {
                        "type": "long"
                    },
                    "store_name": {
                        "type": "keyword"
                    },
                    "country": {
                        "type": "keyword"
                    },
                    "order_type": {
                        "type": "keyword"
                    },
                    "order_type_label": {
                        "type": "keyword"
                    },
                    "created_at": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"
                    },
                    "total_amount": {
                        "type": "double"
                    },
                    "total_items": {
                        "type": "integer"
                    },
                    "items": {
                        "type": "nested",
                        "properties": {
                            "product_id": {"type": "long"},
                            "product_slug": {"type": "keyword"},
                            "quantity": {"type": "integer"},
                            "unit_price": {"type": "double"},
                            "line_total": {"type": "double"}
                        }
                    },
                    "synced_at": {
                        "type": "date",
                        "format": "yyyy-MM-dd HH:mm:ss||epoch_millis"
                    },
                    "day_of_week": {
                        "type": "keyword"
                    },
                    "hour_of_day": {
                        "type": "integer"
                    },
                    "month": {
                        "type": "keyword"
                    }
                }
            }
        }
        
        try:
            # Supprime l'index s'il existe déjà
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                print(f"🗑️  Index existant supprimé: {index_name}")
            
            # Crée le nouvel index
            response = self.client.indices.create(
                index=index_name,
                body=mapping
            )
            
            print(f"✅ Index créé avec succès: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'index: {e}")
            return False
    
    def index_order(self, order_data, index_name='ecommerce_orders'):
        """
        Indexe une commande dans Elasticsearch
        """
        try:
            # Ajoute un timestamp de synchronisation
            order_data['synced_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Ajoute des champs calculés pour l'analyse
            created_at = datetime.strptime(order_data['created_at'], '%Y-%m-%d %H:%M:%S')
            order_data['day_of_week'] = created_at.strftime('%A')
            order_data['hour_of_day'] = created_at.hour
            order_data['month'] = created_at.strftime('%Y-%m')
            
            # Indexe le document
            response = self.client.index(
                index=index_name,
                id=order_data['order_id'],
                body=order_data
            )
            
            logger.info(f"✅ Commande {order_data['order_id']} indexée")
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation: {e}")
            return None
    
    def bulk_index_orders(self, orders_data, index_name='ecommerce_orders'):
        """
        Indexe plusieurs commandes en lot
        """
        print(f"📦 Indexation en lot de {len(orders_data)} commandes...")
        
        try:
            from elasticsearch.helpers import bulk
            
            # Prépare les documents pour l'indexation en lot
            actions = []
            for order in orders_data:
                # Ajoute les champs calculés
                order['synced_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                created_at = datetime.strptime(order['created_at'], '%Y-%m-%d %H:%M:%S')
                order['day_of_week'] = created_at.strftime('%A')
                order['hour_of_day'] = created_at.hour
                order['month'] = created_at.strftime('%Y-%m')
                
                action = {
                    "_index": index_name,
                    "_id": order['order_id'],
                    "_source": order
                }
                actions.append(action)
            
            # Exécute l'indexation en lot
            success, failed = bulk(self.client, actions)
            
            print(f"✅ {success} commandes indexées avec succès")
            if failed:
                print(f"❌ {len(failed)} échecs d'indexation")
            
            return success, failed
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'indexation en lot: {e}")
            return 0, []
    
    def search_orders(self, query=None, index_name='ecommerce_orders', size=10):
        """
        Recherche dans les commandes
        """
        try:
            if query is None:
                # Recherche par défaut - commandes récentes
                search_body = {
                    "query": {"match_all": {}},
                    "sort": [{"created_at": {"order": "desc"}}],
                    "size": size
                }
            else:
                search_body = query
            
            response = self.client.search(
                index=index_name,
                body=search_body
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            return None
    
    def get_analytics(self, index_name='ecommerce_orders'):
        """
        Requêtes d'analyse sur les commandes
        """
        try:
            # Statistiques générales
            stats_query = {
                "size": 0,
                "aggs": {
                    "total_orders": {
                        "value_count": {"field": "order_id"}
                    },
                    "total_amount": {
                        "sum": {"field": "total_amount"}
                    },
                    "avg_order_value": {
                        "avg": {"field": "total_amount"}
                    },
                    "orders_by_country": {
                        "terms": {"field": "country"}
                    },
                    "orders_by_hour": {
                        "terms": {"field": "hour_of_day"}
                    },
                    "orders_by_type": {
                        "terms": {"field": "order_type_label"}
                    }
                }
            }
            
            response = self.client.search(
                index=index_name,
                body=stats_query
            )
            
            return response['aggregations']
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'analyse: {e}")
            return None
    
    def delete_index(self, index_name):
        """
        Supprime un index
        """
        try:
            if self.client.indices.exists(index=index_name):
                self.client.indices.delete(index=index_name)
                print(f"🗑️  Index supprimé: {index_name}")
                return True
            else:
                print(f"ℹ️  Index n'existe pas: {index_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la suppression: {e}")
            return False
    
    def close(self):
        """
        Ferme la connexion
        """
        if self.client:
            logger.info("🔌 Connexion Elasticsearch fermée")

def test_elasticsearch_connection():
    """
    Test de connexion et fonctions de base
    """
    print("🧪 TEST DE CONNEXION ELASTICSEARCH")
    print("=" * 40)
    
    # Test de connexion
    es_client = ElasticsearchClient()
    
    if not es_client.client or not es_client.client.ping():
        print("❌ Elasticsearch n'est pas disponible")
        print("💡 Assurez-vous qu'Elasticsearch est démarré sur localhost:9200")
        return False
    
    # Test de création d'index
    test_index = 'test_ecommerce'
    if es_client.create_orders_index(test_index):
        print(f"✅ Test de création d'index réussi")
        
        # Test d'indexation
        test_order = {
            "order_id": 99999,
            "store_id": 1,
            "store_name": "test-store",
            "country": "France",
            "order_type": 1,
            "order_type_label": "Standard",
            "created_at": "2024-12-01 10:30:00",
            "total_amount": 150.50,
            "total_items": 2,
            "items": [
                {
                    "product_id": 1,
                    "product_slug": "test-product",
                    "quantity": 2,
                    "unit_price": 75.25,
                    "line_total": 150.50
                }
            ]
        }
        
        if es_client.index_order(test_order, test_index):
            print("✅ Test d'indexation réussi")
            
            # Test de recherche
            search_result = es_client.search_orders(index_name=test_index)
            if search_result and search_result['hits']['total']['value'] > 0:
                print("✅ Test de recherche réussi")
            
        # Nettoyage
        es_client.delete_index(test_index)
        print("🧹 Nettoyage terminé")
    
    es_client.close()
    return True

if __name__ == "__main__":
    test_elasticsearch_connection()