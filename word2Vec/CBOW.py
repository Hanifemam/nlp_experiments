class DataLoader():
    def __init__(self, train_set, batch_size=8, shuffle=True):
        self.train_set = train_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        