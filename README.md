# E-Commerce Data Platform

Une plateforme complète d'analyse de données e-commerce avec segmentation IA et optimisation SQL.

## 🚀 Démarrage Rapide

```bash
# Cloner le projet
git clone [<your-repo-url>](https://github.com/zakariaeyahya/ecommerce-data-platform.git)
cd ecommerce-data-platform

# Démarrer avec Docker
docker-compose up --build
```

**Accès aux services :**
- Django : http://localhost:8000
- API REST : http://localhost:5000
- Elasticsearch : http://localhost:9201

## 📋 Fonctionnalités

### ✅ Optimisation SQL
- Requêtes optimisées avec index performants
- Benchmarking automatique des performances
- Amélioration de 13-17% du temps d'exécution

### ✅ Analyse de Cohortes
- Rétention des magasins par semaine
- Visualisations avec heatmaps
- Taux de rétention de 97.5% sur 12 semaines

### ✅ Segmentation IA
- Clustering K-means des utilisateurs
- Embeddings de requêtes de recherche
- 5 segments identifiés (Tech Enthusiasts, Fashion Buyers, etc.)

### ✅ API REST
- Création et recherche de commandes
- Analytics en temps réel
- Synchronisation Elasticsearch

## 🛠️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Elasticsearch  │    │      Redis      │
│   (Données)     │    │   (Recherche)    │    │    (Cache)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
┌─────────────────────────────────┼─────────────────────────────────┐
│                     Application Python                           │
├─────────────────┬───────────────┼───────────────┬─────────────────┤
│   Django Web    │  SQL Optimizer│  Elasticsearch│ Behavioral AI   │
│   (Backend)     │  (Performance)│     Sync      │  (Segmentation) │
├─────────────────┼───────────────┼───────────────┼─────────────────┤
│   Flask API     │ Cohort Analysis│               │                 │
│   (REST)        │ (Retention)    │               │                 │
└─────────────────┴───────────────┴───────────────┴─────────────────┘
```

## 📊 Résultats Générés

Le système génère automatiquement :

- `cohort_analysis_simple.csv` - Données de rétention
- `segmentation_report.json` - Rapport de segmentation IA  
- `user_segments_visualization.png` - Visualisation des segments
- `ml_features.csv` - Features d'apprentissage automatique

## 🔧 Configuration

### Variables d'Environnement

```env
DATABASE_URL=postgresql://postgres:password@postgres:5432/coding_challenge_data
ELASTICSEARCH_URL=http://elasticsearch:9200
REDIS_URL=redis://redis:6379/0
```

### Ports Utilisés

- **8000** : Django Web Interface
- **5000** : Flask REST API
- **9201** : Elasticsearch
- **5433** : PostgreSQL
- **6380** : Redis

## 📈 Métriques de Performance

| Requête | Avant | Après | Amélioration |
|---------|-------|-------|--------------|
| GMV par pays | 69ms | 66ms | +5% |
| Top magasins | 126ms | 109ms | +13% |
| Utilisateurs actifs | 75ms | 62ms | +18% |

## 🧠 Segments IA Identifiés

1. **Tech Enthusiasts** (25%) - Forte affinité technologique
2. **Fashion Buyers** (30%) - Orientés mode et style  
3. **Budget Hunters** (20%) - Sensibles aux prix
4. **Home Improvers** (15%) - Articles de maison
5. **Active Lifestyle** (10%) - Sports et fitness

## 📋 API Endpoints

```bash
# Health check
GET /

# Analytics générales
GET /analytics

# Créer une commande
POST /orders

# Rechercher des commandes
GET /search/orders?country=France&min_amount=100

# Démonstration
POST /demo/create-order
```

## 🛠️ Développement Local

```bash
# Sans Docker
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Démarrer PostgreSQL et Elasticsearch localement
python manage.py migrate
python manage.py generate_data
python manage.py runserver
```

## 📂 Structure du Projet

```
ecommerce-data-platform/
├── behavioral-segmentation/     # Segmentation IA
├── cohort-analyzer/            # Analyse de cohortes
├── elasticsearch-sync/         # Synchronisation ES + API
├── sql-optimizer/             # Optimisation requêtes
├── data_generator/            # Génération de données
├── docker-compose.yml         # Configuration Docker
├── Dockerfile                # Image Docker
├── requirements.txt          # Dépendances Python
└── README.md                # Documentation
```

## 🎯 Résultats de l'Analyse

**Rétention des Magasins :**
- Semaine 1 : 100% (40/40 magasins)
- Semaine 4 : 97.5% (39/40 magasins)  
- Semaine 12 : 97.5% (taux remarquable)

**Répartition Géographique :**
- 8 pays couverts
- 5 magasins par pays (équilibré)
- 14,000+ commandes analysées

## 🤝 Contribution

Ce projet répond aux exigences du challenge Data Engineer YouCan :

1. ✅ **SQL Optimization** - Index et requêtes optimisées
2. ✅ **Cohort Analysis** - Rétention hebdomadaire sur 8 semaines  
3. ✅ **AI Segmentation** - Elasticsearch + embeddings + clustering

## 📄 Licence

Projet développé dans le cadre du challenge technique YouCan.

---

**Développé avec ❤️ pour démontrer l'excellence en ingénierie des données**
