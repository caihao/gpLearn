from bin.automation import Au
from bin.modelInit import initializeModel

au=Au([200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000],[200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000],4000,4000,[4],50,use_data_type="nolimit",use_weight=True,ignore_head_number=0,use_gpu_number=1)
au.load_model("ResNet_200-3000_nolimit_min50_4.pt","ResNet",True)
au.train_step([6],[6e-6],batch_size=1,need_data_info=True)
au.test()
au.finish()
