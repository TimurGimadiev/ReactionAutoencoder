from ReactionAutoencoder import Data, DataGen, NetTrain
import numpy as np
import zipfile
import os
os.chdir("./Data")
with zipfile.ZipFile("dataset.shelve.zip", 'r') as zip_ref:
    zip_ref.extractall("")
data = Data("dataset.shelve")
data.scan()
data.train_test_sets(test_fraction=0.1)
a = NetTrain(data)
a.train_epochs(1)
dg = DataGen(np.random.random((2560, 128)))
res = a.generate(dg)
