import psycopg2
import psycopg2.extras
from datetime import datetime
import time

# Configuration de la base
DB_CONFIG = {
    'host': 'localhost',
    'database': 'coding_challenge_data',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

def get_connection():
    """Connexion à la base de données"""
    return psycopg2.connect(**DB_CONFIG)

def get_gmv_by_country():
    """
    GMV (Gross Merchandise Volume) par pays avec pourcentages
    Calcule le chiffre d'affaires total par pays
    """
    query = """
    SELECT 
        c.name as country_name,
        ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv,
        ROUND(
            (SUM(oi.quantity * p.price) * 100.0 / 
             (SELECT SUM(oi2.quantity * p2.price) 
              FROM order_items oi2 
              JOIN products p2 ON oi2.product_id = p2.id))::numeric, 
            2
        ) as percentage
    FROM countries c
    JOIN stores s ON c.id = s.country_id
    JOIN products p ON s.id = p.store_id
    JOIN order_items oi ON p.id = oi.product_id
    GROUP BY c.id, c.name
    ORDER BY gmv DESC;
    """
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    start_time = time.time()
    cur.execute(query)
    results = cur.fetchall()
    execution_time = time.time() - start_time
    
    cur.close()
    conn.close()
    
    return results, execution_time, query

def get_top_stores_by_gmv(limit=10):
    """
    Top magasins par GMV
    Retourne les magasins avec le plus gros chiffre d'affaires
    """
    query = """
    SELECT 
        s.slug as store_name,
        c.name as country,
        ROUND(SUM(oi.quantity * p.price)::numeric, 2) as gmv,
        COUNT(DISTINCT o.id) as total_orders,
        COUNT(oi.product_id) as total_items_sold
    FROM stores s
    JOIN countries c ON s.country_id = c.id
    JOIN products p ON s.id = p.store_id
    JOIN order_items oi ON p.id = oi.product_id
    JOIN orders o ON oi.order_id = o.id
    GROUP BY s.id, s.slug, c.name
    ORDER BY gmv DESC
    LIMIT %s;
    """
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    start_time = time.time()
    cur.execute(query, (limit,))
    results = cur.fetchall()
    execution_time = time.time() - start_time
    
    cur.close()
    conn.close()
    
    return results, execution_time, query

def get_weekly_active_users():
    """
    Utilisateurs actifs hebdomadaires (approximation avec les magasins)
    Compte les magasins ayant eu des commandes par semaine
    """
    query = """
    SELECT 
        DATE_TRUNC('week', o.created_at) as week_start,
        COUNT(DISTINCT o.store_id) as active_stores,
        COUNT(o.id) as total_orders,
        ROUND(SUM(oi.quantity * p.price)::numeric, 2) as weekly_gmv
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE o.created_at >= '2024-01-01'
    GROUP BY DATE_TRUNC('week', o.created_at)
    ORDER BY week_start;
    """
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    start_time = time.time()
    cur.execute(query)
    results = cur.fetchall()
    execution_time = time.time() - start_time
    
    cur.close()
    conn.close()
    
    return results, execution_time, query

def get_revenue_by_category():
    """
    Revenus par catégorie de produits (estimation basée sur le prix)
    """
    query = """
    SELECT 
        CASE 
            WHEN p.price BETWEEN 50 AND 1500 THEN 'Electronics'
            WHEN p.price BETWEEN 15 AND 200 THEN 'Clothing'
            WHEN p.price BETWEEN 8 AND 45 THEN 'Books'
            WHEN p.price BETWEEN 25 AND 800 THEN 'Home'
            WHEN p.price BETWEEN 20 AND 300 THEN 'Sports'
            ELSE 'Other'
        END as category,
        ROUND(SUM(oi.quantity * p.price)::numeric, 2) as revenue,
        COUNT(oi.product_id) as items_sold,
        ROUND(AVG(p.price)::numeric, 2) as avg_price
    FROM products p
    JOIN order_items oi ON p.id = oi.product_id
    GROUP BY 
        CASE 
            WHEN p.price BETWEEN 50 AND 1500 THEN 'Electronics'
            WHEN p.price BETWEEN 15 AND 200 THEN 'Clothing'
            WHEN p.price BETWEEN 8 AND 45 THEN 'Books'
            WHEN p.price BETWEEN 25 AND 800 THEN 'Home'
            WHEN p.price BETWEEN 20 AND 300 THEN 'Sports'
            ELSE 'Other'
        END
    ORDER BY revenue DESC;
    """
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    start_time = time.time()
    cur.execute(query)
    results = cur.fetchall()
    execution_time = time.time() - start_time
    
    cur.close()
    conn.close()
    
    return results, execution_time, query

def get_monthly_trends():
    """
    Tendances mensuelles des commandes et revenus
    """
    query = """
    SELECT 
        DATE_TRUNC('month', o.created_at) as month,
        COUNT(o.id) as total_orders,
        COUNT(DISTINCT o.store_id) as active_stores,
        ROUND(SUM(oi.quantity * p.price)::numeric, 2) as monthly_revenue,
        ROUND(AVG(oi.quantity * p.price)::numeric, 2) as avg_order_value
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE o.created_at >= '2024-01-01'
    GROUP BY DATE_TRUNC('month', o.created_at)
    ORDER BY month;
    """
    
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    start_time = time.time()
    cur.execute(query)
    results = cur.fetchall()
    execution_time = time.time() - start_time
    
    cur.close()
    conn.close()
    
    return results, execution_time, query

def run_all_queries():
    """
    Exécute toutes les requêtes et affiche les résultats
    """
    print("=== Requêtes SQL - Performance Baseline ===\n")
    
    # 1. GMV par pays
    print("1. GMV par pays:")
    results, exec_time, query = get_gmv_by_country()
    print(f"   Temps d'exécution: {exec_time:.4f}s")
    for row in results:
        print(f"   {row['country_name']}: {row['gmv']}€ ({row['percentage']}%)")
    print()
    
    # 2. Top magasins
    print("2. Top 5 magasins par GMV:")
    results, exec_time, query = get_top_stores_by_gmv(5)
    print(f"   Temps d'exécution: {exec_time:.4f}s")
    for row in results:
        print(f"   {row['store_name']} ({row['country']}): {row['gmv']}€")
    print()
    
    # 3. Revenus par catégorie
    print("3. Revenus par catégorie:")
    results, exec_time, query = get_revenue_by_category()
    print(f"   Temps d'exécution: {exec_time:.4f}s")
    for row in results:
        print(f"   {row['category']}: {row['revenue']}€ ({row['items_sold']} articles)")
    print()
    
    # 4. Tendances mensuelles (résumé)
    print("4. Tendances mensuelles (derniers 3 mois):")
    results, exec_time, query = get_monthly_trends()
    print(f"   Temps d'exécution: {exec_time:.4f}s")
    for row in results[-3:]:  # 3 derniers mois
        month_str = row['month'].strftime('%Y-%m')
        print(f"   {month_str}: {row['total_orders']} commandes, {row['monthly_revenue']}€")
    print()

if __name__ == "__main__":
    run_all_queries()