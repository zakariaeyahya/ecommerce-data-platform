# Configuration pour la generation de donnees

# Nombre d'entites a creer
COUNTRIES_COUNT = 8
STORES_PER_COUNTRY = 5
PRODUCTS_PER_STORE = 20
ORDERS_PER_MONTH = 50  # Par jour en moyenne

# Periode de generation (12 mois)
DATE_RANGE = {
    'start': '2024-01-01',
    'end': '2024-12-31'
}

# Fourchettes de prix par categorie (min, max)
PRICE_RANGES = {
    'electronics': (50.0, 1500.0),
    'clothing': (15.0, 200.0),
    'books': (8.0, 45.0),
    'home': (25.0, 800.0),
    'sports': (20.0, 300.0),
    'default': (10.0, 100.0)
}

# Types de commandes disponibles (correspond aux choix dans le modele Order)
ORDER_TYPES = [1, 2, 3]  # 1=Standard, 2=Express, 3=Premium

# Distribution des types de commandes (probabilites)
ORDER_TYPE_DISTRIBUTION = {
    1: 0.7,  # 70% Standard
    2: 0.2,  # 20% Express
    3: 0.1   # 10% Premium
}

# Configuration avancee pour patterns realistes
SEASONAL_PATTERNS = {
    'high_season': [11, 12],     # Novembre, Decembre (Black Friday, Noël)
    'low_season': [1, 2],        # Janvier, Fevrier (post-holidays)
    'normal_season': [3, 4, 5, 6, 7, 8, 9, 10]
}

# Multiplicateurs saisonniers
SEASONAL_MULTIPLIERS = {
    'high_season': 1.5,
    'low_season': 0.7,
    'normal_season': 1.0
}

# Configuration pour les produits
PRODUCT_CATEGORIES = [
    'electronics',
    'clothing', 
    'books',
    'home',
    'sports'
]

# Distribution des categories par magasin (probabilites)
CATEGORY_DISTRIBUTION = {
    'electronics': 0.25,
    'clothing': 0.30,
    'books': 0.10,
    'home': 0.20,
    'sports': 0.15
}

# Quantites typiques par article de commande
QUANTITY_RANGE = {
    'min': 1,
    'max': 3
}

# Nombre d'articles par commande
ITEMS_PER_ORDER = {
    'min': 1,
    'max': 5
}

# Configuration pour les pays (optionnel - pour personnaliser)
DEFAULT_COUNTRIES = [
    'France',
    'Morocco', 
    'Spain',
    'Italy',
    'Germany',
    'United Kingdom',
    'Netherlands',
    'Belgium'
]

# Configuration de performance pour la generation
BATCH_SIZE = 1000  # Nombre d'objets a creer par batch
COMMIT_FREQUENCY = 500  # Frequence de commit en base

# Configuration de logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'show_progress': True,
    'log_file': 'data_generation.log'
}