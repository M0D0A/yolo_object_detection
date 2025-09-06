class TrainStoper:
    def __init__(self, mode="min", patience=5, threshold=1e-4, threshold_mode="rel"):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None


    def calculation(self, value):
        if self.mode == "min" and self.threshold_mode == "abs":
            return value < self.best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "abs":
            return value > self.best + self.threshold
        elif self.mode == "min" and self.threshold_mode == "rel":
            return value < self.best - self.best * self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return value > self.best + self.best * self.threshold


    def __call__(self, value):
        if self.best is None:
            self.best = value
            return False
        
        if self.calculation(value):
            self.best = value
            self.count = 0
        else:
            self.count+=1

        return self.count >= self.patience
