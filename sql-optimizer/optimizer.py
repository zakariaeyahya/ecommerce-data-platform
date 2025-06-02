import psycopg2
import time
from queries import get_gmv_by_country, get_top_stores_by_gmv, get_weekly_active_users, get_revenue_by_category, DB_CONFIG
from benchmark import measure_query_performance

def get_connection():
    """Connexion à la base de données"""
    return psycopg2.connect(**DB_CONFIG)

def create_performance_indexes_fixed():
    """
    Version corrigée pour créer des index optimaux
    """
    print("=== CRÉATION D'INDEX OPTIMISÉS (VERSION CORRIGÉE) ===\n")
    
    # Index simplifiés sans CONCURRENTLY pour éviter les erreurs de transaction
    optimization_indexes = [
        {
            'name': 'idx_order_items_product_quantity',
            'description': 'Index composite pour optimiser les calculs de GMV',
            'sql': """
            CREATE INDEX IF NOT EXISTS idx_order_items_product_quantity 
            ON order_items (product_id, quantity, order_id);
            """
        },
        {
            'name': 'idx_products_store_price_desc',
            'description': 'Index pour jointures produits-magasins avec prix descendant',
            'sql': """
            CREATE INDEX IF NOT EXISTS idx_products_store_price_desc 
            ON products (store_id, price DESC);
            """
        },
        {
            'name': 'idx_orders_created_store_type',
            'description': 'Index composite pour requêtes temporelles',
            'sql': """
            CREATE INDEX IF NOT EXISTS idx_orders_created_store_type 
            ON orders (created_at, store_id, type) 
            WHERE created_at >= '2024-01-01';
            """
        },
        {
            'name': 'idx_order_items_order_covering',
            'description': 'Index pour couvrir les requêtes order_items',
            'sql': """
            CREATE INDEX IF NOT EXISTS idx_order_items_order_covering 
            ON order_items (order_id, product_id, quantity);
            """
        },
        {
            'name': 'idx_products_price_ranges',
            'description': 'Index pour les fourchettes de prix',
            'sql': """
            CREATE INDEX IF NOT EXISTS idx_products_price_ranges 
            ON products (price, store_id, id);
            """
        }
    ]
    
    conn = get_connection()
    cur = conn.cursor()
    
    for index in optimization_indexes:
        print(f"Création de l'index: {index['name']}")
        print(f"Description: {index['description']}")
        
        try:
            start_time = time.time()
            cur.execute(index['sql'])
            conn.commit()
            execution_time = time.time() - start_time
            
            print(f"✅ Index créé en {execution_time:.2f}s")
        except Exception as e:
            conn.rollback()
            if "already exists" in str(e) or "existe déjà" in str(e):
                print("ℹ️  Index existe déjà")
            else:
                print(f"❌ Erreur: {e}")
        print()
    
    cur.close()
    conn.close()

def create_optimized_views():
    """
    Crée des vues optimisées pour les requêtes fréquentes
    """
    print("=== CRÉATION DE VUES OPTIMISÉES ===\n")
    
    views = [
        {
            'name': 'v_gmv_by_country',
            'description': 'Vue optimisée pour GMV par pays',
            'sql': """
            CREATE OR REPLACE VIEW v_gmv_by_country AS
            SELECT 
                c.id as country_id,
                c.name as country_name,
                ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv,
                COUNT(DISTINCT s.id) as stores_count,
                COUNT(DISTINCT o.id) as orders_count
            FROM countries c
            JOIN stores s ON c.id = s.country_id
            JOIN products p ON s.id = p.store_id
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            GROUP BY c.id, c.name;
            """
        },
        {
            'name': 'v_store_performance',
            'description': 'Vue des performances par magasin',
            'sql': """
            CREATE OR REPLACE VIEW v_store_performance AS
            SELECT 
                s.id as store_id,
                s.slug as store_name,
                c.name as country,
                ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv,
                COUNT(DISTINCT o.id) as total_orders,
                SUM(oi.quantity) as total_items_sold,
                ROUND(AVG(oi.quantity * p.price)::numeric, 2) as avg_order_value,
                COUNT(DISTINCT p.id) as products_count
            FROM stores s
            JOIN countries c ON s.country_id = c.id
            JOIN products p ON s.id = p.store_id
            JOIN order_items oi ON p.id = oi.product_id
            JOIN orders o ON oi.order_id = o.id
            GROUP BY s.id, s.slug, c.name;
            """
        },
        {
            'name': 'v_monthly_trends',
            'description': 'Vue des tendances mensuelles',
            'sql': """
            CREATE OR REPLACE VIEW v_monthly_trends AS
            SELECT 
                DATE_TRUNC('month', o.created_at) as month,
                COUNT(o.id) as total_orders,
                COUNT(DISTINCT o.store_id) as active_stores,
                ROUND(SUM(oi.quantity * p.price)::numeric, 2) as monthly_revenue,
                ROUND(AVG(oi.quantity * p.price)::numeric, 2) as avg_order_value,
                SUM(oi.quantity) as total_items_sold
            FROM orders o
            JOIN order_items oi ON o.id = oi.order_id
            JOIN products p ON oi.product_id = p.id
            WHERE o.created_at >= '2024-01-01'
            GROUP BY DATE_TRUNC('month', o.created_at);
            """
        }
    ]
    
    conn = get_connection()
    cur = conn.cursor()
    
    for view in views:
        print(f"Création de la vue: {view['name']}")
        print(f"Description: {view['description']}")
        
        try:
            start_time = time.time()
            cur.execute(view['sql'])
            conn.commit()
            execution_time = time.time() - start_time
            
            print(f"✅ Vue créée en {execution_time:.3f}s")
        except Exception as e:
            conn.rollback()
            print(f"❌ Erreur: {e}")
        print()
    
    cur.close()
    conn.close()

def test_optimized_views():
    """
    Teste les performances des vues optimisées
    """
    print("=== TEST DES VUES OPTIMISÉES ===\n")
    
    test_queries = [
        {
            'name': 'GMV par pays (vue)',
            'sql': """
            SELECT 
                country_name,
                gmv,
                ROUND((gmv * 100.0 / SUM(gmv) OVER ())::numeric, 2) as percentage
            FROM v_gmv_by_country 
            ORDER BY gmv DESC;
            """
        },
        {
            'name': 'Top 5 magasins (vue)',
            'sql': """
            SELECT 
                store_name,
                country,
                gmv,
                total_orders,
                total_items_sold
            FROM v_store_performance 
            ORDER BY gmv DESC 
            LIMIT 5;
            """
        },
        {
            'name': 'Tendances mensuelles (vue)',
            'sql': """
            SELECT 
                to_char(month, 'YYYY-MM') as month_str,
                total_orders,
                monthly_revenue,
                avg_order_value
            FROM v_monthly_trends 
            ORDER BY month DESC 
            LIMIT 6;
            """
        }
    ]
    
    conn = get_connection()
    cur = conn.cursor()
    
    for query in test_queries:
        print(f"Test: {query['name']}")
        try:
            start_time = time.time()
            cur.execute(query['sql'])
            results = cur.fetchall()
            execution_time = time.time() - start_time
            
            print(f"✅ Exécuté en {execution_time:.4f}s ({len(results)} résultats)")
            
            # Affiche les premiers résultats
            for i, row in enumerate(results[:3]):
                print(f"   {row}")
            if len(results) > 3:
                print(f"   ... et {len(results) - 3} autres")
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
        print()
    
    cur.close()
    conn.close()

def benchmark_with_fixes():
    """
    Nouveau benchmark avec les corrections
    """
    print("=== BENCHMARK AVEC OPTIMISATIONS CORRIGÉES ===\n")
    
    # Fonctions à tester
    query_functions = [
        ('GMV par pays', get_gmv_by_country, []),
        ('Top magasins', get_top_stores_by_gmv, [10]),
        ('Utilisateurs actifs', get_weekly_active_users, []),
        ('Revenus par catégorie', get_revenue_by_category, [])
    ]
    
    print("Performance AVANT nouvelles optimisations:")
    print("-" * 50)
    
    before_results = []
    for name, func, args in query_functions:
        stats, _, _ = measure_query_performance(func, 2, *args)
        if stats:
            before_results.append((name, stats['avg_time']))
            print(f"{name}: {stats['avg_time']:.4f}s")
    
    print("\nApplication des optimisations corrigées...")
    create_performance_indexes_fixed()
    create_optimized_views()
    
    print("Performance APRÈS nouvelles optimisations:")
    print("-" * 50)
    
    after_results = []
    for name, func, args in query_functions:
        stats, _, _ = measure_query_performance(func, 2, *args)
        if stats:
            after_results.append((name, stats['avg_time']))
            print(f"{name}: {stats['avg_time']:.4f}s")
    
    print("\n=== RÉSUMÉ FINAL DES AMÉLIORATIONS ===")
    print(f"{'Requête':<20} {'Avant':<10} {'Après':<10} {'Amélioration':<15}")
    print("-" * 60)
    
    for i, (name, before_time) in enumerate(before_results):
        if i < len(after_results):
            _, after_time = after_results[i]
            improvement = ((before_time - after_time) / before_time) * 100
            improvement_str = f"{improvement:+.1f}%"
            
            print(f"{name:<20} {before_time:<10.4f} {after_time:<10.4f} {improvement_str:<15}")
    
    print("\nTest des vues optimisées:")
    test_optimized_views()

def main():
    """
    Optimisation complète corrigée
    """
    print("🚀 OPTIMISEUR SQL CORRIGÉ - E-COMMERCE DATA PLATFORM\n")
    
    benchmark_with_fixes()
    
    print("✅ Optimisation corrigée terminée!")
    print("\n🎯 Prochaine étape: Analyse de cohortes pour étudier la rétention des magasins")

if __name__ == "__main__":
    main()