from bin.automation import Au

au=Au([150],[],7500,4000,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=4,use_data_type="nonoise_3_full",centering=True,use_weight=True,train_type="angle",need_data_info=False)
au.load_model("ResAngleNet_nonoise_150-1000.pt","ResAngleNet",True)
au.train_step([10],[1e-6])
au.test()
au.finish()
