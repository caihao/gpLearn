from bin.automation import Au

au=Au([200,300,400],[300,400,500],100,100,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="nolimit_new",centering=True,use_weight=True,train_type="particle",need_data_info=True)
au.load_model("ResNet_custom_test.pt","ResNet_custom",True)
au.train_step([2],[2e-6])
au.test()
au.finish()
