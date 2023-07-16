from bin.automation import Au

au=Au([100,150,200],[],100,100,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="nolimit_3",centering=True,use_weight=True,train_type="position",need_data_info=True)
au.load_model("PointNet_test.pt","PointNet",True)
au.train_step([1,1],[6e-6,2e-6])
au.test()
au.finish()
