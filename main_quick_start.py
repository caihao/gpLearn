from bin.automation import Au
from bin.modelInit import initializeModel

au=Au([500,1000,1500,2000,2500,3000],[2500,1000,1500,2000,2500,3000],2500,2500,[4],50,use_data_type="nolimit",use_weight=True)
au.load_model("ResNet_test.pt","ResNet",True)
au.train_step([5])
au.test()
au.finish()
