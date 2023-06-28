from bin.automation import Au

au=Au([60,70,80,90,100,200,300,400,500,600,700,800],[60,70,80,90,100,200,300,400,500,600,700,800],5000,5000,[4],allow_min_pix_number=True,ignore_head_number=0,batch_size=2,use_data_type="nolimit_new",centering=True,use_weight=True)
au.load_model("ResNet_custom_test_5000.pt","ResNet_custom",True)
au.train_step([4,2,1],[8e-6,6e-6,4e-6],need_data_info=True)
au.test()
au.finish()
