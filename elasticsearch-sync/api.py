from flask import Flask, request, jsonify
from datetime import datetime
import psycopg2
import psycopg2.extras
from es_client import ElasticsearchClient
import logging
import random

# Configuration
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configuration de la base de données
DB_CONFIG = {
    'host': 'localhost',
    'database': 'coding_challenge_data',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

# Client Elasticsearch global
es_client = ElasticsearchClient()

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Connexion à la base de données"""
    return psycopg2.connect(**DB_CONFIG)

@app.route('/', methods=['GET'])
def health_check():
    """
    Health check de l'API
    """
    return jsonify({
        "status": "healthy",
        "service": "E-Commerce Data Platform API",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "elasticsearch": es_client.client.ping() if es_client.client else False
    })

@app.route('/orders', methods=['POST'])
def create_order():
    """
    Crée une nouvelle commande et la synchronise avec Elasticsearch
    """
    try:
        data = request.get_json()
        
        # Validation des données requises
        required_fields = ['store_id', 'items']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Champ requis manquant: {field}"}), 400
        
        # Récupère les détails du magasin
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT s.id, s.slug, c.name as country 
            FROM stores s 
            JOIN countries c ON s.country_id = c.id 
            WHERE s.id = %s
        """, (data['store_id'],))
        
        store = cur.fetchone()
        if not store:
            return jsonify({"error": "Magasin introuvable"}), 404
        
        # Crée la commande
        order_type = data.get('type', 1)
        created_at = datetime.now()
        
        cur.execute("""
            INSERT INTO orders (type, store_id, created_at) 
            VALUES (%s, %s, %s) RETURNING id
        """, (order_type, data['store_id'], created_at))
        
        order_id = cur.fetchone()['id']
        
        # Ajoute les articles
        total_amount = 0
        total_items = 0
        order_items = []
        
        for item in data['items']:
            if 'product_id' not in item or 'quantity' not in item:
                return jsonify({"error": "Article invalide: product_id et quantity requis"}), 400
            
            # Récupère le produit
            cur.execute("""
                SELECT id, slug, price, store_id 
                FROM products 
                WHERE id = %s AND store_id = %s
            """, (item['product_id'], data['store_id']))
            
            product = cur.fetchone()
            if not product:
                return jsonify({"error": f"Produit {item['product_id']} introuvable dans ce magasin"}), 404
            
            quantity = int(item['quantity'])
            line_total = quantity * float(product['price'])
            
            # Insère l'article
            cur.execute("""
                INSERT INTO order_items (order_id, product_id, quantity) 
                VALUES (%s, %s, %s)
            """, (order_id, item['product_id'], quantity))
            
            # Accumule les totaux
            total_amount += line_total
            total_items += quantity
            
            order_items.append({
                'product_id': product['id'],
                'product_slug': product['slug'],
                'quantity': quantity,
                'unit_price': float(product['price']),
                'line_total': round(line_total, 2)
            })
        
        conn.commit()
        
        # Prépare les données pour Elasticsearch
        order_data = {
            'order_id': order_id,
            'store_id': data['store_id'],
            'store_name': store['slug'],
            'country': store['country'],
            'order_type': order_type,
            'order_type_label': {1: 'Standard', 2: 'Express', 3: 'Premium'}.get(order_type, 'Unknown'),
            'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'total_amount': round(total_amount, 2),
            'total_items': total_items,
            'items': order_items
        }
        
        # Synchronise avec Elasticsearch
        es_result = es_client.index_order(order_data, 'ecommerce_orders')
        
        cur.close()
        conn.close()
        
        # Réponse
        response = {
            'success': True,
            'order_id': order_id,
            'total_amount': round(total_amount, 2),
            'total_items': total_items,
            'elasticsearch_synced': es_result is not None,
            'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info(f"✅ Commande {order_id} créée et synchronisée")
        return jsonify(response), 201
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création: {e}")
        return jsonify({"error": "Erreur serveur", "details": str(e)}), 500

@app.route('/orders/<int:order_id>', methods=['GET'])
def get_order(order_id):
    """
    Récupère une commande par son ID
    """
    try:
        # Recherche dans Elasticsearch d'abord
        search_query = {
            "query": {"term": {"order_id": order_id}},
            "size": 1
        }
        
        es_result = es_client.search_orders(search_query, 'ecommerce_orders')
        
        if es_result and es_result['hits']['total']['value'] > 0:
            order_data = es_result['hits']['hits'][0]['_source']
            return jsonify({
                'success': True,
                'source': 'elasticsearch',
                'order': order_data
            })
        
        # Fallback: recherche dans PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                o.id as order_id,
                o.type as order_type,
                o.created_at,
                o.store_id,
                s.slug as store_name,
                c.name as country
            FROM orders o
            JOIN stores s ON o.store_id = s.id
            JOIN countries c ON s.country_id = c.id
            WHERE o.id = %s
        """, (order_id,))
        
        order = cur.fetchone()
        if not order:
            return jsonify({"error": "Commande introuvable"}), 404
        
        cur.close()
        conn.close()
        
        # Convertit en format JSON
        order_data = dict(order)
        order_data['created_at'] = order['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'source': 'postgresql',
            'order': order_data
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération: {e}")
        return jsonify({"error": "Erreur serveur", "details": str(e)}), 500

@app.route('/search/orders', methods=['GET'])
def search_orders():
    """
    Recherche des commandes avec filtres
    """
    try:
        # Paramètres de recherche
        country = request.args.get('country')
        order_type = request.args.get('order_type')
        min_amount = request.args.get('min_amount', type=float)
        max_amount = request.args.get('max_amount', type=float)
        limit = request.args.get('limit', 10, type=int)
        
        # Construction de la requête Elasticsearch
        must_clauses = []
        
        if country:
            must_clauses.append({"term": {"country": country}})
        
        if order_type:
            must_clauses.append({"term": {"order_type_label": order_type}})
        
        if min_amount or max_amount:
            range_clause = {"range": {"total_amount": {}}}
            if min_amount:
                range_clause["range"]["total_amount"]["gte"] = min_amount
            if max_amount:
                range_clause["range"]["total_amount"]["lte"] = max_amount
            must_clauses.append(range_clause)
        
        # Requête finale
        if must_clauses:
            search_query = {
                "query": {"bool": {"must": must_clauses}},
                "sort": [{"created_at": {"order": "desc"}}],
                "size": limit
            }
        else:
            search_query = {
                "query": {"match_all": {}},
                "sort": [{"created_at": {"order": "desc"}}],
                "size": limit
            }
        
        # Exécute la recherche
        es_result = es_client.search_orders(search_query, 'ecommerce_orders')
        
        if es_result:
            orders = [hit['_source'] for hit in es_result['hits']['hits']]
            total = es_result['hits']['total']['value']
            
            return jsonify({
                'success': True,
                'total': total,
                'count': len(orders),
                'orders': orders,
                'filters_applied': {
                    'country': country,
                    'order_type': order_type,
                    'min_amount': min_amount,
                    'max_amount': max_amount
                }
            })
        else:
            return jsonify({'success': False, 'error': 'Erreur de recherche'}), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la recherche: {e}")
        return jsonify({"error": "Erreur serveur", "details": str(e)}), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """
    Récupère les analytics des commandes
    """
    try:
        analytics = es_client.get_analytics('ecommerce_orders')
        
        if analytics:
            # Formate les résultats
            response = {
                'success': True,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'summary': {
                    'total_orders': analytics['total_orders']['value'],
                    'total_revenue': round(analytics['total_amount']['value'], 2),
                    'average_order_value': round(analytics['avg_order_value']['value'], 2)
                },
                'breakdown': {
                    'by_country': [
                        {'country': bucket['key'], 'orders': bucket['doc_count']}
                        for bucket in analytics['orders_by_country']['buckets']
                    ],
                    'by_order_type': [
                        {'type': bucket['key'], 'orders': bucket['doc_count']}
                        for bucket in analytics['orders_by_type']['buckets']
                    ],
                    'by_hour': [
                        {'hour': bucket['key'], 'orders': bucket['doc_count']}
                        for bucket in analytics['orders_by_hour']['buckets']
                    ]
                }
            }
            
            return jsonify(response)
        else:
            return jsonify({'success': False, 'error': 'Données analytics indisponibles'}), 500
            
    except Exception as e:
        logger.error(f"❌ Erreur analytics: {e}")
        return jsonify({"error": "Erreur serveur", "details": str(e)}), 500

@app.route('/demo/create-order', methods=['POST'])
def demo_create_order():
    """
    Crée une commande de démonstration
    """
    try:
        # Récupère un magasin aléatoire
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("SELECT id FROM stores ORDER BY RANDOM() LIMIT 1")
        store = cur.fetchone()
        
        if not store:
            return jsonify({"error": "Aucun magasin disponible"}), 404
        
        store_id = store['id']
        
        # Récupère des produits aléatoires de ce magasin
        cur.execute("""
            SELECT id FROM products 
            WHERE store_id = %s 
            ORDER BY RANDOM() 
            LIMIT %s
        """, (store_id, random.randint(1, 3)))
        
        products = cur.fetchall()
        cur.close()
        conn.close()
        
        # Crée les articles
        items = []
        for product in products:
            items.append({
                'product_id': product['id'],
                'quantity': random.randint(1, 3)
            })
        
        # Données de la commande
        order_data = {
            'store_id': store_id,
            'type': random.choice([1, 2, 3]),
            'items': items
        }
        
        # Appelle l'endpoint de création
        with app.test_client() as client:
            response = client.post('/orders', 
                                 json=order_data,
                                 content_type='application/json')
            
            if response.status_code == 201:
                result = response.get_json()
                return jsonify({
                    'success': True,
                    'message': 'Commande de démonstration créée',
                    'order': result
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Échec de création de la commande démo'
                }), 500
                
    except Exception as e:
        logger.error(f"❌ Erreur démonstration: {e}")
        return jsonify({"error": "Erreur serveur", "details": str(e)}), 500

if __name__ == '__main__':
    print("🚀 LANCEMENT DE L'API E-COMMERCE")
    print("=" * 40)
    print("📍 Endpoints disponibles:")
    print("   GET  /                    - Health check")
    print("   POST /orders              - Créer une commande")
    print("   GET  /orders/<id>         - Récupérer une commande")
    print("   GET  /search/orders       - Rechercher des commandes")
    print("   GET  /analytics           - Analytics des commandes")
    print("   POST /demo/create-order   - Créer une commande démo")
    print()
    print("🔍 Exemples d'utilisation:")
    print("   curl http://localhost:5000/")
    print("   curl http://localhost:5000/analytics")
    print("   curl -X POST http://localhost:5000/demo/create-order")
    print()
    print("🌐 Interface disponible sur: http://localhost:5000")
    print("=" * 40)
    
    app.run(host='0.0.0.0', port=5000, debug=True)