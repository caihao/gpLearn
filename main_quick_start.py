from bin.automation import Au

au=Au([150,200,250,300,400,500,600,800,1000],[450,600,750,900,1200,1500,1800,2400,3000],100,100,[4],limit_min_pix_number=True,ignore_head_number=0,batch_size=1,use_data_type="demo",centering=True,use_weight=True,train_type="particle",log_name="LOG_NAME_DEMO",need_data_info=True,current_file_name="main_quick_start.py")
au.load_model("MODULE_NAME_DEMO",modelInit=True)
au.train_step([4,2],[6e-6,2e-6])
au.test()
au.finish()
