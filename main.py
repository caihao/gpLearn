from bin.automation import Au

au=Au([60,70,80,90,100,200,300,400,500,600,700,800],[60,70,80,90,100,200,300,400,500,600,700,800],2500,7000,[4],allow_min_pix_number=20,ignore_head_number=0,use_data_type="nolimit_new",centering=True,use_weight=True,use_gpu_number=1)
au.load_model("PointNet_test.pt","PointNet",True)
au.train_step([3,3,3],[8e-6,2e-6,8e-7],batch_size=1,need_data_info=True)
au.test()
au.finish()
