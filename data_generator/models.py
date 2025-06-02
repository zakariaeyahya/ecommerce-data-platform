from django.db import models
from django.utils import timezone


class Country(models.Model):
    """Modele pour les pays"""
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    class Meta:
        db_table = 'countries'
        verbose_name = 'Country'
        verbose_name_plural = 'Countries'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class Store(models.Model):
    """Modele pour les magasins"""
    id = models.AutoField(primary_key=True)
    slug = models.SlugField(max_length=255, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    country = models.ForeignKey(
        Country, 
        on_delete=models.CASCADE,
        related_name='stores'
    )
    
    class Meta:
        db_table = 'stores'
        verbose_name = 'Store'
        verbose_name_plural = 'Stores'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['country', 'created_at']),
            models.Index(fields=['slug']),
        ]
    
    def __str__(self):
        return f"{self.slug} ({self.country.name})"


class Product(models.Model):
    """Modele pour les produits"""
    id = models.AutoField(primary_key=True)
    slug = models.SlugField(max_length=255)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    store = models.ForeignKey(
        Store,
        on_delete=models.CASCADE,
        related_name='products'
    )
    
    class Meta:
        db_table = 'products'
        verbose_name = 'Product'
        verbose_name_plural = 'Products'
        ordering = ['slug']
        indexes = [
            models.Index(fields=['store', 'price']),
            models.Index(fields=['slug']),
        ]
    
    def __str__(self):
        return f"{self.slug} - {self.price}€"


class Order(models.Model):
    """Modele pour les commandes"""
    ORDER_TYPE_CHOICES = [
        (1, 'Standard'),
        (2, 'Express'),
        (3, 'Premium'),
    ]
    
    id = models.AutoField(primary_key=True)
    type = models.IntegerField(choices=ORDER_TYPE_CHOICES, default=1)
    created_at = models.DateTimeField(default=timezone.now)
    store = models.ForeignKey(
        Store,
        on_delete=models.CASCADE,
        related_name='orders'
    )
    
    class Meta:
        db_table = 'orders'
        verbose_name = 'Order'
        verbose_name_plural = 'Orders'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['store', 'created_at']),
            models.Index(fields=['created_at']),
            models.Index(fields=['type']),
        ]
    
    def __str__(self):
        return f"Order #{self.id} - {self.get_type_display()}"
    
    @property
    def total_amount(self):
        """Calcule le montant total de la commande"""
        return sum(
            item.quantity * item.product.price 
            for item in self.order_items.all()
        )


class OrderItem(models.Model):
    """Modele pour les articles de commande"""
    order = models.ForeignKey(
        Order,
        on_delete=models.CASCADE,
        related_name='order_items'
    )
    product = models.ForeignKey(
        Product,
        on_delete=models.CASCADE,
        related_name='order_items'
    )
    quantity = models.IntegerField(default=1)
    
    class Meta:
        db_table = 'order_items'
        verbose_name = 'Order Item'
        verbose_name_plural = 'Order Items'
        unique_together = ('order', 'product')
        indexes = [
            models.Index(fields=['order', 'product']),
            models.Index(fields=['product']),
        ]
    
    def __str__(self):
        return f"{self.product.slug} x{self.quantity}"
    
    @property
    def line_total(self):
        """Calcule le total de cette ligne"""
        return self.quantity * self.product.price