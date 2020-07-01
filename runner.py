from NeuralNetwork import *

imrunningouttanames = NeuralNetwork("Imrunningouttanames")
imrunningouttanames.extract("Imrunningouttanames")

obadiah = Trainer(imrunningouttanames)
imrunningouttanames = obadiah.biotrain(1)
imrunningouttanames = obadiah.trainbc(1)
imrunningouttanames.save()