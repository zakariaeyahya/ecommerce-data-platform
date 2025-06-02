import os
import sys
import django
from datetime import datetime, timedelta
from faker import Faker
import random
import argparse

# Ajouter le répertoire parent au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Configuration Django explicite
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ecommerce_platform.settings')
django.setup()

from data_generator.models import Country, Store, Product, Order, OrderItem
from data_generator.config import *

fake = Faker()

def generate_countries():
    """Crée les pays de base"""
    print("Génération des pays...")
    
    countries_data = [
        'France', 'Morocco', 'Spain', 'Italy', 'Germany', 
        'United Kingdom', 'Netherlands', 'Belgium'
    ]
    
    countries = []
    for name in countries_data:
        country, created = Country.objects.get_or_create(
            name=name,
            defaults={'created_at': fake.date_time_between(start_date='-2y', end_date='now')}
        )
        countries.append(country)
        if created:
            print(f"Pays créé: {name}")
    
    return countries

def generate_stores(countries):
    """Génere les magasins avec distribution géographique"""
    print("Génération des magasins...")
    
    stores = []
    for country in countries:
        for i in range(STORES_PER_COUNTRY):
            store = Store.objects.create(
                slug=f"store-{country.name.lower().replace(' ', '-')}-{i+1}",
                country=country,
                created_at=fake.date_time_between(start_date='-1y', end_date='-6M')
            )
            stores.append(store)
    
    print(f"{len(stores)} magasins créés")
    return stores

def generate_products(stores):
    """Crée les produits par magasin avec prix réalistes"""
    print("Génération des produits...")
    
    products = []
    categories = ['electronics', 'clothing', 'books', 'home', 'sports']
    
    for store in stores:
        for i in range(PRODUCTS_PER_STORE):
            category = random.choice(categories)
            price_range = PRICE_RANGES.get(category, (10, 100))
            
            product = Product.objects.create(
                slug=f"product-{store.id}-{i+1}",
                price=round(random.uniform(*price_range), 2),
                store=store
            )
            products.append(product)
    
    print(f"{len(products)} produits créés")
    return products

def generate_orders(stores):
    """Génere les commandes avec patterns saisonniers"""
    print("Génération des commandes...")
    
    orders = []
    start_date = datetime.strptime(DATE_RANGE['start'], '%Y-%m-%d')
    end_date = datetime.strptime(DATE_RANGE['end'], '%Y-%m-%d')
    
    current_date = start_date
    while current_date <= end_date:
        # Pattern saisonnier
        month = current_date.month
        if month in SEASONAL_PATTERNS['high_season']:
            daily_orders = int(ORDERS_PER_MONTH * SEASONAL_MULTIPLIERS['high_season'])
        elif month in SEASONAL_PATTERNS['low_season']:
            daily_orders = int(ORDERS_PER_MONTH * SEASONAL_MULTIPLIERS['low_season'])
        else:
            daily_orders = int(ORDERS_PER_MONTH * SEASONAL_MULTIPLIERS['normal_season'])
        
        # Génere les commandes pour ce jour
        for _ in range(random.randint(daily_orders//2, daily_orders)):
            store = random.choice(stores)
            order_type = random.choices(
                ORDER_TYPES,
                weights=[ORDER_TYPE_DISTRIBUTION[t] for t in ORDER_TYPES]
            )[0]
            
            order = Order.objects.create(
                type=order_type,
                store=store,
                created_at=current_date + timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
            )
            orders.append(order)
        
        current_date += timedelta(days=1)
    
    print(f"{len(orders)} commandes créées")
    return orders

def generate_order_items(orders):
    """Crée les articles de commande"""
    print("Génération des articles de commande...")
    
    order_items = []
    for order in orders:
        # Récupere les produits du magasin de cette commande
        store_products = list(Product.objects.filter(store=order.store))
        
        if store_products:
            # Nombre d'articles par commande
            num_items = random.randint(
                ITEMS_PER_ORDER['min'],
                min(ITEMS_PER_ORDER['max'], len(store_products))
            )
            selected_products = random.sample(store_products, num_items)
            
            for product in selected_products:
                quantity = random.randint(
                    QUANTITY_RANGE['min'],
                    QUANTITY_RANGE['max']
                )
                
                order_item = OrderItem.objects.create(
                    order=order,
                    product=product,
                    quantity=quantity
                )
                order_items.append(order_item)
    
    print(f"{len(order_items)} articles de commande créés")
    return order_items

def populate_database():
    """Fonction principale d'orchestration"""
    print("=== Début de la génération de données ===")
    
    # Génere les données dans l'ordre des dépendances
    countries = generate_countries()
    stores = generate_stores(countries)
    products = generate_products(stores)
    orders = generate_orders(stores)
    order_items = generate_order_items(orders)
    
    print("=== Génération terminée ===")
    print(f"Résumé:")
    print(f"- {len(countries)} pays")
    print(f"- {len(stores)} magasins")
    print(f"- {len(products)} produits")
    print(f"- {len(orders)} commandes")
    print(f"- {len(order_items)} articles de commande")

def main():
    """Point d'entrée avec parametres CLI"""
    parser = argparse.ArgumentParser(description='Générateur de données pour le challenge')
    parser.add_argument('--reset', action='store_true', 
                       help='Supprime toutes les données existantes')
    parser.add_argument('--countries', type=int, default=COUNTRIES_COUNT,
                       help='Nombre de pays a créer')
    
    args = parser.parse_args()
    
    if args.reset:
        print("Suppression des données existantes...")
        OrderItem.objects.all().delete()
        Order.objects.all().delete()
        Product.objects.all().delete()
        Store.objects.all().delete()
        Country.objects.all().delete()
        print("Données supprimées")
    
    # Lance la génération
    populate_database()

if __name__ == "__main__":
    main()