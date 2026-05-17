"""
NepalMed AI: Medical Inventory Tracker
Tracks essential drugs and diagnostic kits at the local and referral level.
"""

import json
import os

class StockManager:
    def __init__(self, stock_path="data/inventory/stock_levels.json"):
        self.stock_path = stock_path
        os.makedirs(os.path.dirname(self.stock_path), exist_ok=True)
        self.essential_items = [
            "rK39 Diagnostic Kit",
            "Liposomal Amphotericin B (L-AmB)",
            "Dengue RDT Kit",
            "Paracetamol (Syrup)",
            "Iron/Folic Acid Tablets"
        ]
        
        if not os.path.exists(self.stock_path):
            self._initialize_stock()

    def _initialize_stock(self):
        initial_stock = {item: 100 for item in self.essential_items}
        with open(self.stock_path, "w") as f:
            json.dump(initial_stock, f, indent=4)

    def get_stock(self):
        with open(self.stock_path, "r") as f:
            return json.load(f)

    def update_stock(self, item, change):
        stock = self.get_stock()
        if item in stock:
            stock[item] = max(0, stock[item] + change)
            with open(self.stock_path, "w") as f:
                json.dump(stock, f, indent=4)
            return True
        return False

    def check_availability(self, item):
        stock = self.get_stock()
        level = stock.get(item, 0)
        if level <= 0:
            return "OUT OF STOCK"
        if level < 10:
            return "CRITICAL"
        return "AVAILABLE"

if __name__ == "__main__":
    sm = StockManager()
    print(sm.get_stock())
