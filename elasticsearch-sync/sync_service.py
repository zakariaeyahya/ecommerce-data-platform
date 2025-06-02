import psycopg2
import psycopg2.extras
from es_client import ElasticsearchClient
from datetime import datetime
import time
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la base de données
DB_CONFIG = {
    'host': 'localhost',
    'database': 'coding_challenge_data',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

class OrderSyncService:
    """
    Service de synchronisation des commandes PostgreSQL vers Elasticsearch
    """
    
    def __init__(self):
        """
        Initialise le service de synchronisation
        """
        self.es_client = ElasticsearchClient()
        self.db_conn = None
        self.index_name = 'ecommerce_orders'
        
    def connect_database(self):
        """
        Établit la connexion à la base de données
        """
        try:
            self.db_conn = psycopg2.connect(**DB_CONFIG)
            logger.info("✅ Connexion PostgreSQL établie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur de connexion PostgreSQL: {e}")
            return False
    
    def setup_elasticsearch_index(self):
        """
        Configure l'index Elasticsearch pour les commandes
        """
        print("🔧 Configuration de l'index Elasticsearch...")
        
        if self.es_client.create_orders_index(self.index_name):
            print(f"✅ Index {self.index_name} configuré")
            return True
        else:
            print(f"❌ Échec de configuration de l'index {self.index_name}")
            return False
    
    def extract_orders_from_db(self, limit=None, offset=0):
        """
        Extrait les commandes de PostgreSQL avec toutes les informations nécessaires
        """
        query = """
        SELECT 
            o.id as order_id,
            o.type as order_type,
            CASE 
                WHEN o.type = 1 THEN 'Standard'
                WHEN o.type = 2 THEN 'Express'
                WHEN o.type = 3 THEN 'Premium'
                ELSE 'Unknown'
            END as order_type_label,
            o.created_at,
            o.store_id,
            s.slug as store_name,
            c.name as country,
            
            -- Calculs des totaux
            ROUND(SUM(oi.quantity * p.price)::numeric, 2) as total_amount,
            SUM(oi.quantity) as total_items,
            
            -- Articles de la commande (format JSON)
            JSON_AGG(
                JSON_BUILD_OBJECT(
                    'product_id', p.id,
                    'product_slug', p.slug,
                    'quantity', oi.quantity,
                    'unit_price', p.price,
                    'line_total', ROUND((oi.quantity * p.price)::numeric, 2)
                )
            ) as items
            
        FROM orders o
        JOIN stores s ON o.store_id = s.id
        JOIN countries c ON s.country_id = c.id
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        GROUP BY o.id, o.type, o.created_at, o.store_id, s.slug, c.name
        ORDER BY o.created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        try:
            cur = self.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query)
            orders = cur.fetchall()
            cur.close()
            
            # Convertit les résultats en format compatible Elasticsearch
            formatted_orders = []
            for order in orders:
                formatted_order = dict(order)
                # Convertit la date en string
                formatted_order['created_at'] = order['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                formatted_orders.append(formatted_order)
            
            return formatted_orders
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction: {e}")
            return []
    
    def sync_single_order(self, order_id):
        """
        Synchronise une commande spécifique
        """
        logger.info(f"🔄 Synchronisation de la commande {order_id}")
        
        # Extrait la commande spécifique
        query = """
        SELECT 
            o.id as order_id,
            o.type as order_type,
            CASE 
                WHEN o.type = 1 THEN 'Standard'
                WHEN o.type = 2 THEN 'Express'
                WHEN o.type = 3 THEN 'Premium'
                ELSE 'Unknown'
            END as order_type_label,
            o.created_at,
            o.store_id,
            s.slug as store_name,
            c.name as country,
            ROUND(SUM(oi.quantity * p.price)::numeric, 2) as total_amount,
            SUM(oi.quantity) as total_items,
            JSON_AGG(
                JSON_BUILD_OBJECT(
                    'product_id', p.id,
                    'product_slug', p.slug,
                    'quantity', oi.quantity,
                    'unit_price', p.price,
                    'line_total', ROUND((oi.quantity * p.price)::numeric, 2)
                )
            ) as items
        FROM orders o
        JOIN stores s ON o.store_id = s.id
        JOIN countries c ON s.country_id = c.id
        JOIN order_items oi ON o.id = oi.order_id
        JOIN products p ON oi.product_id = p.id
        WHERE o.id = %s
        GROUP BY o.id, o.type, o.created_at, o.store_id, s.slug, c.name;
        """
        
        try:
            cur = self.db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(query, (order_id,))
            order = cur.fetchone()
            cur.close()
            
            if order:
                # Formate la commande
                formatted_order = dict(order)
                formatted_order['created_at'] = order['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Indexe dans Elasticsearch
                result = self.es_client.index_order(formatted_order, self.index_name)
                
                if result:
                    logger.info(f"✅ Commande {order_id} synchronisée")
                    return True
                else:
                    logger.error(f"❌ Échec de synchronisation de la commande {order_id}")
                    return False
            else:
                logger.warning(f"⚠️  Commande {order_id} introuvable")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation: {e}")
            return False
    
    def full_sync(self, batch_size=1000):
        """
        Synchronisation complète de toutes les commandes
        """
        print("🔄 SYNCHRONISATION COMPLÈTE")
        print("=" * 30)
        
        if not self.connect_database():
            return False
        
        if not self.setup_elasticsearch_index():
            return False
        
        # Compte le nombre total de commandes
        cur = self.db_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM orders")
        total_orders = cur.fetchone()[0]
        cur.close()
        
        print(f"📊 {total_orders} commandes à synchroniser")
        
        # Synchronisation par batches
        synced_count = 0
        offset = 0
        
        while offset < total_orders:
            print(f"🔄 Batch {offset//batch_size + 1}: commandes {offset+1} à {min(offset+batch_size, total_orders)}")
            
            # Extrait le batch
            orders = self.extract_orders_from_db(limit=batch_size, offset=offset)
            
            if orders:
                # Synchronise le batch
                success, failed = self.es_client.bulk_index_orders(orders, self.index_name)
                synced_count += success
                
                if failed:
                    logger.warning(f"⚠️  {len(failed)} échecs dans ce batch")
            
            offset += batch_size
            time.sleep(0.1)  # Pause pour éviter la surcharge
        
        print(f"✅ Synchronisation terminée: {synced_count}/{total_orders} commandes")
        return True
    
    def get_sync_status(self):
        """
        Vérifie le statut de synchronisation
        """
        try:
            # Compte dans PostgreSQL
            cur = self.db_conn.cursor()
            cur.execute("SELECT COUNT(*) FROM orders")
            db_count = cur.fetchone()[0]
            cur.close()
            
            # Compte dans Elasticsearch
            es_response = self.es_client.search_orders(
                query={"query": {"match_all": {}}, "size": 0},
                index_name=self.index_name
            )
            
            es_count = 0
            if es_response:
                es_count = es_response['hits']['total']['value']
            
            print(f"📊 STATUT DE SYNCHRONISATION:")
            print(f"   PostgreSQL: {db_count} commandes")
            print(f"   Elasticsearch: {es_count} commandes")
            print(f"   Synchronisation: {es_count/db_count*100:.1f}%" if db_count > 0 else "N/A")
            
            return db_count, es_count
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification: {e}")
            return 0, 0
    
    def test_search_capabilities(self):
        """
        Teste les capacités de recherche Elasticsearch
        """
        print("\n🔍 TEST DES CAPACITÉS DE RECHERCHE")
        print("=" * 40)
        
        # 1. Recherche des commandes récentes
        print("1. Commandes récentes:")
        recent_orders = self.es_client.search_orders(size=5)
        if recent_orders:
            for hit in recent_orders['hits']['hits']:
                order = hit['_source']
                print(f"   Commande #{order['order_id']}: {order['total_amount']}€ ({order['country']})")
        
        # 2. Recherche par pays
        print("\n2. Commandes par pays (France):")
        country_query = {
            "query": {"term": {"country": "France"}},
            "size": 3
        }
        france_orders = self.es_client.search_orders(country_query)
        if france_orders:
            for hit in france_orders['hits']['hits']:
                order = hit['_source']
                print(f"   Commande #{order['order_id']}: {order['total_amount']}€")
        
        # 3. Analytics
        print("\n3. Analytics:")
        analytics = self.es_client.get_analytics()
        if analytics:
            print(f"   Total commandes: {analytics['total_orders']['value']}")
            print(f"   CA total: {analytics['total_amount']['value']:.2f}€")
            print(f"   Panier moyen: {analytics['avg_order_value']['value']:.2f}€")
            
            print("   Top pays:")
            for bucket in analytics['orders_by_country']['buckets'][:3]:
                print(f"     {bucket['key']}: {bucket['doc_count']} commandes")
    
    def close_connections(self):
        """
        Ferme toutes les connexions
        """
        if self.db_conn:
            self.db_conn.close()
            logger.info("🔌 Connexion PostgreSQL fermée")
        
        if self.es_client:
            self.es_client.close()

def main():
    """
    Lance le service de synchronisation complet
    """
    print("🚀 SERVICE DE SYNCHRONISATION ELASTICSEARCH")
    print("=" * 50)
    
    sync_service = OrderSyncService()
    
    try:
        # 1. Synchronisation complète
        if sync_service.full_sync():
            
            # 2. Vérification du statut
            sync_service.get_sync_status()
            
            # 3. Test des fonctionnalités de recherche
            sync_service.test_search_capabilities()
            
            print("\n✅ Service de synchronisation configuré avec succès!")
            print("\n🎯 Prochaine étape: API Flask pour les nouvelles commandes")
        
    except KeyboardInterrupt:
        print("\n🛑 Synchronisation interrompue par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur dans le service: {e}")
    finally:
        sync_service.close_connections()

if __name__ == "__main__":
    main()