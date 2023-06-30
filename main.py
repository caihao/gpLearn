from bin.automation import Au

au=Au([50,60,70,80,90,100,200,300,400,500,600,700],[50,60,70,80,90,100,200,300,400,500,600,700],3000,7000,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="nolimit_new",centering=True,use_weight=True)
au.load_model("ResNet_custom_conv1024.pt","ResNet_custom",True)
au.train_step([4,3],[6e-6,3e-6],need_data_info=True)
au.test()
au.finish()
