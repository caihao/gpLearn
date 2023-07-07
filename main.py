from bin.automation import Au

au=Au([30,40,50,60,80,100,150,200,250,300,400,500,600,800,1000,1500,2000,2500,3000,4000,5000],[90,120,150,180,240,300,450,600,750,900,1200,1500,1800,2400,3000,4500,6000,7500,9000,12000,15000],5000,8000,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="nolimit_3",centering=True,use_weight=True,train_type="particle",need_data_info=True)
au.load_model("ResNet_custom_3_30-5000.pt","ResNet_custom",True)
au.train_step([6,4,4],[2e-6,4e-6,6e-6])
au.test()
au.finish()
