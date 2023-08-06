from bin.automation import Au

au=Au([150,200,250,300,400,500,600,800,1000],[],6000,4000,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="nolimit_3_full",centering=True,use_weight=True,train_type="angle",need_data_info=True)
au.load_model("ResAngleNet_150-1000_min20.pt","ResAngleNet",True)
au.train_step([9],[1e-6])
au.test()
au.finish()