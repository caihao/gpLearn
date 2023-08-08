from bin.automation import Au

au=Au([150,200,250,300,400,500,600,800,1000],[],10000,4000,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=4,use_data_type="nonoise_3_full",centering=True,use_weight=True,train_type="angle",need_data_info=False)
au.load_model("ResAngleNet_nonoise_150-1000_gpu4.pt","ResAngleNet",True)
au.train_step([5,6],[1e-6,6e-7])
au.test()
au.finish()
