import numpy as np

class PerishableInventoryEnv:

    def __init__(self,
                 shelf_life=5,
                 demand_mean=12,
                 demand_var=0.3,
                 hold_cost=0.1,
                 spoil_penalty=1.5):

        self.shelf_life = shelf_life
        self.demand_mean = demand_mean
        self.demand_var = demand_var
        self.hold_cost = hold_cost
        self.spoil_penalty = spoil_penalty

        self.action_space = [0, 5, 10, 20]
        self.reset()

    def reset(self):
        self.inventory = []
        for d in range(1, self.shelf_life + 1):
            qty = np.random.randint(1, 5)
            self.inventory.append([qty, d])
        return self.get_state()

    def sample_demand(self):
        std = self.demand_mean * self.demand_var
        return max(1, int(np.random.normal(self.demand_mean, std)))

    def get_state(self):
        age_buckets = np.zeros(self.shelf_life)

        for qty, age in self.inventory:
            age_buckets[age-1] += qty

        age_buckets /= 20.0
        demand = self.sample_demand() / 50.0

        return np.concatenate([age_buckets, [demand]])

    def step(self, action_id):

        order_qty = self.action_space[action_id]
        demand = self.sample_demand()

        expired = 0

        # age inventory
        for item in self.inventory:
            item[1] -= 1

        new_inventory = []
        for qty, age in self.inventory:
            if age <= 0:
                expired += qty
            else:
                new_inventory.append([qty, age])
        self.inventory = new_inventory

        # add order
        if order_qty > 0:
            self.inventory.append([order_qty, self.shelf_life])

        # FIFO selling
        self.inventory.sort(key=lambda x: x[1])

        sold = 0
        remaining = demand

        for item in self.inventory:
            take = min(item[0], remaining)
            item[0] -= take
            sold += take
            remaining -= take

        self.inventory = [i for i in self.inventory if i[0] > 0]

        total_inventory = sum(i[0] for i in self.inventory)

        sales = sold * 2.5
        hold_cost = total_inventory * self.hold_cost
        spoil = expired * self.spoil_penalty
        stockout = max(demand - sold, 0) * 0.5

        reward = sales - hold_cost - spoil - stockout

        info = {
            "sales": sales,
            "holding": hold_cost,
            "spoil": spoil,
            "stockout": stockout
        }

        return self.get_state(), reward, False, info
