import psycopg2
import pandas as pd
from datetime import datetime

# Configuration de la base
DB_CONFIG = {
    'host': 'localhost',
    'database': 'coding_challenge_data',
    'user': 'postgres',
    'password': 'password',
    'port': '5432'
}

def simple_cohort_analysis():
    """
    Analyse de cohortes simplifiée sans gestion des fuseaux horaires
    """
    print("🚀 ANALYSE DE COHORTES SIMPLIFIÉE")
    print("=" * 40)
    
    # Requête SQL simplifiée
    query = """
    WITH store_cohorts AS (
        SELECT 
            s.id as store_id,
            s.slug as store_name,
            c.name as country,
            DATE_TRUNC('month', s.created_at)::date as cohort_month,
            MIN(o.created_at)::date as first_order_date
        FROM stores s
        JOIN countries c ON s.country_id = c.id
        LEFT JOIN orders o ON s.id = o.store_id
        GROUP BY s.id, s.slug, c.name, DATE_TRUNC('month', s.created_at)
    ),
    weekly_activity AS (
        SELECT 
            s.id as store_id,
            DATE_TRUNC('week', o.created_at)::date as activity_week,
            COUNT(o.id) as orders_count
        FROM stores s
        JOIN orders o ON s.id = o.store_id
        WHERE o.created_at >= '2024-01-01'
        GROUP BY s.id, DATE_TRUNC('week', o.created_at)
    )
    SELECT 
        sc.store_id,
        sc.store_name,
        sc.country,
        sc.cohort_month,
        sc.first_order_date,
        wa.activity_week,
        wa.orders_count,
        CASE 
            WHEN sc.first_order_date IS NOT NULL AND wa.activity_week IS NOT NULL
            THEN (wa.activity_week - DATE_TRUNC('week', sc.first_order_date)::date) / 7
            ELSE NULL
        END as week_number
    FROM store_cohorts sc
    LEFT JOIN weekly_activity wa ON sc.store_id = wa.store_id
    ORDER BY sc.cohort_month, sc.store_id, wa.activity_week;
    """
    
    try:
        # Connexion et extraction
        conn = psycopg2.connect(**DB_CONFIG)
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"📊 {len(df)} enregistrements extraits")
        
        # Analyse des magasins actifs
        active_stores = df[df['first_order_date'].notna()]
        total_stores = len(active_stores['store_id'].unique())
        
        print(f"🏪 {total_stores} magasins actifs trouvés")
        
        # Analyse par pays
        country_stats = active_stores.groupby('country').agg({
            'store_id': 'nunique',
            'orders_count': 'sum'
        }).round(0)
        
        print("\n🌍 RÉPARTITION PAR PAYS:")
        for country, stats in country_stats.iterrows():
            print(f"   {country}: {int(stats['store_id'])} magasins, {int(stats['orders_count'])} commandes")
        
        # Analyse par cohorte mensuelle
        monthly_cohorts = active_stores.groupby('cohort_month').agg({
            'store_id': 'nunique'
        })
        
        print("\n📅 NOUVEAUX MAGASINS PAR MOIS:")
        for month, stats in monthly_cohorts.iterrows():
            print(f"   {month}: {int(stats['store_id'])} nouveaux magasins")
        
        # Calcul de rétention simple
        activity_with_weeks = df[(df['week_number'].notna()) & (df['week_number'] >= 0) & (df['week_number'] <= 12)]
        
        if not activity_with_weeks.empty:
            print("\n📈 RÉTENTION PAR SEMAINE:")
            
            retention_by_week = activity_with_weeks.groupby('week_number')['store_id'].nunique()
            total_cohort_size = len(active_stores['store_id'].unique())
            
            for week, active_count in retention_by_week.head(8).items():
                retention_rate = active_count / total_cohort_size
                print(f"   Semaine {int(week)}: {active_count} magasins actifs ({retention_rate:.1%})")
        
        # Insights simples
        print("\n💡 INSIGHTS PRINCIPAUX:")
        
        if total_stores > 30:
            print("   ✅ Bonne adoption de la plateforme")
        else:
            print("   ⚠️  Adoption modérée de la plateforme")
        
        # Répartition équilibrée
        country_distribution = country_stats['store_id']
        if country_distribution.std() < country_distribution.mean() * 0.5:
            print("   ✅ Répartition géographique équilibrée")
        else:
            print("   ⚠️  Concentration géographique notable")
        
        # Recommandations
        print("\n📋 RECOMMANDATIONS:")
        print("   • Analyser les magasins inactifs pour comprendre les freins")
        print("   • Mettre en place un programme d'onboarding pour nouveaux magasins")
        print("   • Développer des incitations pour maintenir l'activité")
        
        # Export des données
        df.to_csv('cohort_analysis_simple.csv', index=False)
        print(f"\n💾 Données exportées: cohort_analysis_simple.csv")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_visualization(df):
    """
    Visualisation simple avec matplotlib
    """
    try:
        import matplotlib.pyplot as plt
        
        print("\n🎨 Création de visualisations simples...")
        
        # Graphique 1: Magasins par pays
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        active_stores = df[df['first_order_date'].notna()]
        country_counts = active_stores.groupby('country')['store_id'].nunique()
        
        plt.bar(country_counts.index, country_counts.values, color='skyblue')
        plt.title('Magasins actifs par pays')
        plt.xlabel('Pays')
        plt.ylabel('Nombre de magasins')
        plt.xticks(rotation=45)
        
        # Graphique 2: Nouveaux magasins par mois
        plt.subplot(1, 2, 2)
        monthly_cohorts = active_stores.groupby('cohort_month')['store_id'].nunique()
        
        plt.plot(monthly_cohorts.index, monthly_cohorts.values, marker='o', linewidth=2)
        plt.title('Nouveaux magasins par mois')
        plt.xlabel('Mois de création')
        plt.ylabel('Nouveaux magasins')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('simple_cohort_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Graphique sauvegardé: simple_cohort_analysis.png")
        
        plt.show()
        
    except ImportError:
        print("⚠️  Matplotlib non disponible, pas de visualisation")
    except Exception as e:
        print(f"❌ Erreur visualisation: {e}")

def main():
    """
    Lance l'analyse simplifiée
    """
    df = simple_cohort_analysis()
    
    if df is not None:
        create_simple_visualization(df)
        print("\n✅ Analyse de cohortes simplifiée terminée!")
        print("\n🎯 Prochaine étape: Configuration d'Elasticsearch pour le monitoring temps réel")
    
    return df

if __name__ == "__main__":
    main()