from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
from django.utils import timezone
from faker import Faker
import random
import logging

from data_generator.models import Country, Store, Product, Order, OrderItem
from data_generator.config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

class Command(BaseCommand):
    help = 'Genère des donnees de test pour le challenge'

    def add_arguments(self, parser):
        logger.info("Adding command arguments")
        parser.add_argument(
            '--reset',
            action='store_true',
            help='Supprime toutes les donnees existantes',
        )
        parser.add_argument(
            '--countries',
            type=int,
            default=COUNTRIES_COUNT,
            help='Nombre de pays à creer',
        )

    def handle(self, *args, **options):
        logger.info("Starting data generation handle")
        if options['reset']:
            self.stdout.write("Suppression des donnees existantes...")
            OrderItem.objects.all().delete()
            Order.objects.all().delete()
            Product.objects.all().delete()
            Store.objects.all().delete()
            Country.objects.all().delete()
            self.stdout.write(self.style.SUCCESS("Donnees supprimees"))

        self.stdout.write("=== Debut de la generation de donnees ===")

        # Genère les donnees dans l'ordre des dependances
        countries = self.generate_countries()
        stores = self.generate_stores(countries)
        products = self.generate_products(stores)
        orders = self.generate_orders(stores)
        order_items = self.generate_order_items(orders)

        self.stdout.write("=== Generation terminee ===")
        self.stdout.write(f"Resume:")
        self.stdout.write(f"- {len(countries)} pays")
        self.stdout.write(f"- {len(stores)} magasins")
        self.stdout.write(f"- {len(products)} produits")
        self.stdout.write(f"- {len(orders)} commandes")
        self.stdout.write(f"- {len(order_items)} articles de commande")

    def generate_countries(self):
        logger.info("Generating countries")
        self.stdout.write("Generation des pays...")

        countries_data = DEFAULT_COUNTRIES[:COUNTRIES_COUNT]

        countries = []
        for name in countries_data:
            country, created = Country.objects.get_or_create(
                name=name,
                defaults={'created_at': timezone.make_aware(fake.date_time_between(start_date='-2y', end_date='now'))}
            )
            countries.append(country)
            if created:
                self.stdout.write(f"Pays cree: {name}")

        return countries

    def generate_stores(self, countries):
        logger.info("Generating stores")
        self.stdout.write("Generation des magasins...")

        stores = []
        for country in countries:
            for i in range(STORES_PER_COUNTRY):
                # Fix: remplacer les espaces par des tirets dans le slug
                country_slug = country.name.lower().replace(' ', '-')
                store = Store.objects.create(
                    slug=f"store-{country_slug}-{i+1}",
                    country=country,
                    created_at=timezone.make_aware(fake.date_time_between(start_date='-1y', end_date='-6M'))
                )
                stores.append(store)

        self.stdout.write(f"{len(stores)} magasins crees")
        return stores

    def generate_products(self, stores):
        logger.info("Generating products")
        self.stdout.write("Generation des produits...")

        products = []

        for store in stores:
            for i in range(PRODUCTS_PER_STORE):
                category = random.choice(PRODUCT_CATEGORIES)
                price_range = PRICE_RANGES.get(category, PRICE_RANGES['default'])

                product = Product.objects.create(
                    slug=f"product-{store.id}-{i+1}",
                    price=round(random.uniform(*price_range), 2),
                    store=store
                )
                products.append(product)

        self.stdout.write(f"{len(products)} produits crees")
        return products

    def generate_orders(self, stores):
        logger.info("Generating orders")
        self.stdout.write("Generation des commandes...")

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

            # Genère les commandes pour ce jour
            for _ in range(random.randint(daily_orders//2, daily_orders)):
                store = random.choice(stores)
                order_type = random.choices(
                    ORDER_TYPES,
                    weights=[ORDER_TYPE_DISTRIBUTION[t] for t in ORDER_TYPES]
                )[0]

                order = Order.objects.create(
                    type=order_type,
                    store=store,
                    created_at=timezone.make_aware(current_date + timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                    ))
                )
                orders.append(order)

            current_date += timedelta(days=1)

        self.stdout.write(f"{len(orders)} commandes creees")
        return orders

    def generate_order_items(self, orders):
        logger.info("Generating order items")
        self.stdout.write("Generation des articles de commande...")

        order_items = []
        for order in orders:
            # Recupère les produits du magasin de cette commande
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

        self.stdout.write(f"{len(order_items)} articles de commande crees")
        return order_items