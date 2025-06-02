import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('default')
sns.set_palette("viridis")

def create_retention_heatmap(cohort_table_pct, save_path='cohort_heatmap.png'):
    """
    Crée une heatmap des taux de rétention
    """
    print("🎨 Création de la heatmap de rétention...")
    
    if cohort_table_pct is None or cohort_table_pct.empty:
        print("❌ Pas de données pour la heatmap")
        return None
    
    # Configuration de la figure
    plt.figure(figsize=(15, 8))
    
    # Prépare les données pour la heatmap
    heatmap_data = cohort_table_pct.copy()
    
    # Formate les labels des cohortes
    heatmap_data.index = heatmap_data.index.strftime('%Y-%m-%d')
    
    # Crée la heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1%',
        cmap='RdYlGn',
        center=0.5,
        square=False,
        cbar_kws={'label': 'Taux de rétention'},
        linewidths=0.5
    )
    
    # Personnalisation
    plt.title('Heatmap de rétention des magasins par cohorte\n(% de magasins actifs par semaine)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Semaines depuis la première commande', fontsize=12)
    plt.ylabel('Cohorte (semaine de première commande)', fontsize=12)
    
    # Rotation des labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Ajuste la mise en page
    plt.tight_layout()
    
    # Sauvegarde
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap sauvegardée: {save_path}")
    
    plt.show()
    return ax

def plot_cohort_trends(cohort_table_pct, save_path='cohort_trends.png'):
    """
    Graphique des tendances de rétention par cohorte
    """
    print("📈 Création du graphique de tendances...")
    
    if cohort_table_pct is None or cohort_table_pct.empty:
        print("❌ Pas de données pour les tendances")
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Calcule la rétention moyenne par semaine
    avg_retention = cohort_table_pct.mean()
    
    # Graphique principal - courbe moyenne
    plt.subplot(2, 1, 1)
    plt.plot(avg_retention.index, avg_retention.values, 
             marker='o', linewidth=3, markersize=8, color='darkblue')
    plt.title('Rétention moyenne des magasins par semaine', fontsize=14, fontweight='bold')
    plt.xlabel('Semaines depuis la première commande')
    plt.ylabel('Taux de rétention moyen')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Ajoute des annotations pour les points clés
    if len(avg_retention) > 0:
        week_0 = avg_retention.iloc[0] if len(avg_retention) > 0 else 0
        week_4 = avg_retention.iloc[4] if len(avg_retention) > 4 else 0
        week_8 = avg_retention.iloc[8] if len(avg_retention) > 8 else 0
        
        if week_0 > 0:
            plt.annotate(f'{week_0:.1%}', (0, week_0), 
                        xytext=(0, week_0 + 0.05), fontsize=10, ha='center')
        if week_4 > 0:
            plt.annotate(f'{week_4:.1%}', (4, week_4), 
                        xytext=(4, week_4 + 0.05), fontsize=10, ha='center')
        if week_8 > 0:
            plt.annotate(f'{week_8:.1%}', (8, week_8), 
                        xytext=(8, week_8 + 0.05), fontsize=10, ha='center')
    
    # Graphique secondaire - cohortes individuelles
    plt.subplot(2, 1, 2)
    for i, (cohort_date, retention_data) in enumerate(cohort_table_pct.iterrows()):
        if i < 5:  # Limite à 5 cohortes pour la lisibilité
            plt.plot(retention_data.index, retention_data.values, 
                    marker='o', alpha=0.7, linewidth=2,
                    label=cohort_date.strftime('%Y-%m-%d'))
    
    plt.title('Rétention par cohorte individuelle (top 5)', fontsize=14, fontweight='bold')
    plt.xlabel('Semaines depuis la première commande')
    plt.ylabel('Taux de rétention')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique de tendances sauvegardé: {save_path}")
    
    plt.show()

def create_cohort_size_chart(cohort_table, save_path='cohort_sizes.png'):
    """
    Graphique des tailles de cohortes
    """
    print("📊 Création du graphique des tailles de cohortes...")
    
    if cohort_table is None or cohort_table.empty:
        print("❌ Pas de données pour les tailles de cohortes")
        return None
    
    plt.figure(figsize=(12, 6))
    
    # Calcule la taille de chaque cohorte (colonne 0 = semaine de démarrage)
    if 0 in cohort_table.columns:
        cohort_sizes = cohort_table[0]
        
        # Graphique en barres
        bars = plt.bar(range(len(cohort_sizes)), cohort_sizes.values, 
                      color='skyblue', alpha=0.8, edgecolor='navy')
        
        # Personnalisation
        plt.title('Taille des cohortes de magasins\n(Nombre de nouveaux magasins par période)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Cohorte (par ordre chronologique)')
        plt.ylabel('Nombre de magasins')
        
        # Labels des dates sur l'axe X
        plt.xticks(range(len(cohort_sizes)), 
                  [date.strftime('%m/%d') for date in cohort_sizes.index], 
                  rotation=45)
        
        # Ajoute les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Graphique des tailles sauvegardé: {save_path}")
        
        plt.show()

def create_retention_funnel(cohort_table_pct, save_path='retention_funnel.png'):
    """
    Crée un graphique en entonnoir de rétention
    """
    print("🔄 Création de l'entonnoir de rétention...")
    
    if cohort_table_pct is None or cohort_table_pct.empty:
        print("❌ Pas de données pour l'entonnoir")
        return None
    
    plt.figure(figsize=(10, 8))
    
    # Calcule la rétention moyenne par étape
    avg_retention = cohort_table_pct.mean()
    
    # Prend les étapes clés (semaines 0, 1, 2, 4, 8, 12)
    key_weeks = [0, 1, 2, 4, 8]
    key_weeks = [w for w in key_weeks if w in avg_retention.index]
    
    if len(key_weeks) < 2:
        print("❌ Pas assez de données pour l'entonnoir")
        return None
    
    # Données pour l'entonnoir
    retention_values = [avg_retention[w] for w in key_weeks]
    week_labels = [f'Semaine {w}' for w in key_weeks]
    
    # Couleurs dégradées
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(retention_values)))
    
    # Création de l'entonnoir horizontal
    y_positions = range(len(retention_values))
    bars = plt.barh(y_positions, retention_values, color=colors, alpha=0.8)
    
    # Personnalisation
    plt.title('Entonnoir de rétention des magasins', fontsize=16, fontweight='bold')
    plt.xlabel('Taux de rétention')
    plt.yticks(y_positions, week_labels)
    
    # Ajoute les pourcentages sur les barres
    for i, (bar, value) in enumerate(zip(bars, retention_values)):
        plt.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.1%}', va='center', fontweight='bold')
    
    # Ligne de base à 100%
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='100%')
    
    plt.xlim(0, 1.1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Entonnoir de rétention sauvegardé: {save_path}")
    
    plt.show()

def generate_all_visualizations(cohort_table, cohort_table_pct):
    """
    Génère toutes les visualisations en une fois
    """
    print("🎨 GÉNÉRATION DE TOUTES LES VISUALISATIONS")
    print("=" * 50)
    
    if cohort_table is None or cohort_table_pct is None:
        print("❌ Données manquantes pour les visualisations")
        return
    
    # 1. Heatmap de rétention
    create_retention_heatmap(cohort_table_pct, 'cohort_retention_heatmap.png')
    
    # 2. Tendances de rétention
    plot_cohort_trends(cohort_table_pct, 'cohort_retention_trends.png')
    
    # 3. Tailles des cohortes
    create_cohort_size_chart(cohort_table, 'cohort_sizes.png')
    
    # 4. Entonnoir de rétention
    create_retention_funnel(cohort_table_pct, 'retention_funnel.png')
    
    print("\n✅ Toutes les visualisations générées!")
    print("📁 Fichiers créés:")
    print("   • cohort_retention_heatmap.png")
    print("   • cohort_retention_trends.png") 
    print("   • cohort_sizes.png")
    print("   • retention_funnel.png")

def main():
    """
    Lance la visualisation des cohortes
    """
    print("🎨 VISUALISEUR DE COHORTES")
    print("=" * 30)
    
    try:
        # Importe les données depuis l'analyse
        from cohort_analysis import main as run_cohort_analysis
        
        print("Lancement de l'analyse de cohortes...")
        cohort_table, cohort_table_pct, metrics = run_cohort_analysis()
        
        if cohort_table is not None and cohort_table_pct is not None:
            print("\nGénération des visualisations...")
            generate_all_visualizations(cohort_table, cohort_table_pct)
        else:
            print("❌ Pas de données à visualiser")
    
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()