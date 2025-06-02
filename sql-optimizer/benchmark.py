import psycopg2
import time
import statistics
from queries import get_gmv_by_country, get_top_stores_by_gmv, get_weekly_active_users, get_revenue_by_category, DB_CONFIG

def measure_query_performance(query_func, runs=5, *args):
    """
    Mesure les performances d'une requête sur plusieurs exécutions
    """
    execution_times = []
    results = None
    query = None
    
    print(f"Mesure de performance pour {query_func.__name__} ({runs} exécutions)...")
    
    for i in range(runs):
        try:
            result, exec_time, query_text = query_func(*args)
            execution_times.append(exec_time)
            if results is None:  # Garde le premier résultat
                results = result
                query = query_text
            print(f"  Exécution {i+1}: {exec_time:.4f}s")
        except Exception as e:
            print(f"  Erreur lors de l'exécution {i+1}: {e}")
    
    if execution_times:
        avg_time = statistics.mean(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        performance_stats = {
            'function_name': query_func.__name__,
            'avg_time': avg_time,
            'min_time': min_time,
            'max_time': max_time,
            'total_runs': len(execution_times),
            'results_count': len(results) if results else 0
        }
        
        print(f"  Temps moyen: {avg_time:.4f}s")
        print(f"  Temps min/max: {min_time:.4f}s / {max_time:.4f}s")
        print(f"  Nombre de résultats: {len(results) if results else 0}")
        print()
        
        return performance_stats, results, query
    
    return None, None, None

def analyze_query_plans():
    """
    Analyse les plans d'exécution des requêtes critiques
    """
    print("=== Analyse des plans d'exécution ===\n")
    
    # Requêtes à analyser
    queries_to_analyze = [
        {
            'name': 'GMV par pays',
            'query': """
            EXPLAIN ANALYZE
            SELECT 
                c.name as country_name,
                ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv
            FROM countries c
            JOIN stores s ON c.id = s.country_id
            JOIN products p ON s.id = p.store_id
            JOIN order_items oi ON p.id = oi.product_id
            GROUP BY c.id, c.name
            ORDER BY gmv DESC;
            """
        },
        {
            'name': 'Top magasins',
            'query': """
            EXPLAIN ANALYZE
            SELECT 
                s.slug as store_name,
                ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv
            FROM stores s
            JOIN products p ON s.id = p.store_id
            JOIN order_items oi ON p.id = oi.product_id
            GROUP BY s.id, s.slug
            ORDER BY gmv DESC
            LIMIT 10;
            """
        }
    ]
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    for query_info in queries_to_analyze:
        print(f"Plan d'exécution - {query_info['name']}:")
        print("-" * 50)
        
        try:
            cur.execute(query_info['query'])
            plan = cur.fetchall()
            
            for line in plan:
                print(f"  {line[0]}")
            print()
            
        except Exception as e:
            print(f"Erreur lors de l'analyse: {e}")
            print()
    
    cur.close()
    conn.close()

def check_existing_indexes():
    """
    Vérifie les index existants sur les tables
    """
    print("=== Index existants ===\n")
    
    query = """
    SELECT 
        schemaname,
        tablename,
        indexname,
        indexdef
    FROM pg_indexes 
    WHERE schemaname = 'public'
    AND tablename IN ('countries', 'stores', 'products', 'orders', 'order_items')
    ORDER BY tablename, indexname;
    """
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    try:
        cur.execute(query)
        indexes = cur.fetchall()
        
        current_table = None
        for schema, table, index_name, index_def in indexes:
            if table != current_table:
                if current_table is not None:
                    print()
                print(f"Table: {table}")
                current_table = table
            
            print(f"  - {index_name}")
            print(f"    {index_def}")
        
    except Exception as e:
        print(f"Erreur lors de la vérification des index: {e}")
    
    cur.close()
    conn.close()

def get_table_statistics():
    """
    Statistiques sur les tables pour comprendre la distribution des données
    """
    print("=== Statistiques des tables ===\n")
    
    stats_queries = [
        {
            'name': 'Répartition des commandes par mois',
            'query': """
            SELECT 
                DATE_TRUNC('month', created_at) as month,
                COUNT(*) as orders_count
            FROM orders 
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY month;
            """
        },
        {
            'name': 'Répartition des produits par magasin',
            'query': """
            SELECT 
                COUNT(*) as products_per_store,
                COUNT(DISTINCT store_id) as stores_count
            FROM products 
            GROUP BY store_id
            LIMIT 5;
            """
        },
        {
            'name': 'Types de commandes',
            'query': """
            SELECT 
                type,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM orders
            GROUP BY type
            ORDER BY type;
            """
        }
    ]
    
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    for stat in stats_queries:
        print(f"{stat['name']}:")
        try:
            cur.execute(stat['query'])
            results = cur.fetchall()
            
            for row in results:
                print(f"  {row}")
        except Exception as e:
            print(f"  Erreur: {e}")
        print()
    
    cur.close()
    conn.close()

def run_full_benchmark():
    """
    Benchmark complet de toutes les requêtes
    """
    print("=== BENCHMARK COMPLET DES REQUÊTES SQL ===\n")
    
    # Liste des fonctions à benchmarker
    query_functions = [
        (get_gmv_by_country, []),
        (get_top_stores_by_gmv, [10]),
        (get_weekly_active_users, []),
        (get_revenue_by_category, [])
    ]
    
    benchmark_results = []
    
    # Benchmark de chaque requête
    for query_func, args in query_functions:
        stats, results, query = measure_query_performance(query_func, 3, *args)
        if stats:
            benchmark_results.append(stats)
    
    # Résumé des performances
    print("=== RÉSUMÉ DES PERFORMANCES ===")
    print(f"{'Requête':<25} {'Temps moyen':<12} {'Résultats':<10}")
    print("-" * 50)
    
    for stats in benchmark_results:
        print(f"{stats['function_name']:<25} {stats['avg_time']:<12.4f} {stats['results_count']:<10}")
    
    print()
    
    # Analyse complémentaire
    check_existing_indexes()
    print()
    get_table_statistics()
    print()
    analyze_query_plans()

if __name__ == "__main__":
    run_full_benchmark()