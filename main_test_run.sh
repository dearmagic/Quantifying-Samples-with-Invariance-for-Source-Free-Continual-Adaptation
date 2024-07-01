# ## 3,1,2
# #ok
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source clipart --target painting --model_name final_domainnet --txt clipart2painting.txt \
#  > ./log/main_test_clipart2painting.log 2>&1 & 
#  wait
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source clipart --target real --model_name final_domainnet --txt clipart2real.txt --dataset domainnet/real \
#  > ./log/main_test_clipart2real.log 2>&1 &
# #ok
# wait
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target painting --model_name final_domainnet --txt real2painting.txt \
#  > ./log/main_test_real2painting.log 2>&1 &
# #runnning
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target sketch --model_name final_domainnet --txt real2sketch.txt --dataset domainnet/sketch \
#  > ./log/main_test_real2sketch.log 2>&1 &


# ## 1,2,3
# #  #--deviceid 1 2 3
# #ok
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source clipart --target sketch --model_name final_domainnet --txt clipart2sketch.txt --dataset domainnet/sketch \
#  > ./log/main_test_clipart2sketch.log 2>&1 &
# #ok
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_painting_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source painting --target clipart --model_name final_domainnet --txt painting2clipart.txt --dataset domainnet/clipart \
#  > ./log/main_test_paint2clipart.log 2>&1 &
#  #--deviceid 1 2 3
# #running 
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_sketch_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240502-1024-OH_sketch_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source sketch --target clipart --model_name final_domainnet --txt sketch2clipart.txt --dataset domainnet/clipart \
#  > ./log/main_test_sketch2clipart.log 2>&1 &
#  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_sketch_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240502-1024-OH_sketch_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source sketch --target painting --model_name final_domainnet --txt sketch2painting.txt \
#  > ./log/main_test_sketch2painting.log 2>&1 & 

# ## 2,1,3
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test_copy.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_painting_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source painting --target real --model_name final_domainnet --txt painting2real.txt --dataset domainnet/real \
#  > ./log/main_test_paint2real_onetask.log 2>&1 &
#  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test_copy_.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_painting_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source painting --target real --model_name final_domainnet --txt painting2real_.txt --dataset domainnet/real \
#  > ./log/main_test_paint2real_all.log 2>&1 &
# #ok
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_painting_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_painting_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source painting --target sketch --model_name final_domainnet --txt painting2sketch.txt --dataset domainnet/sketch \
#  > ./log/main_test_paint2sketch.log 2>&1 &
# #ok
#  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target clipart --model_name final_domainnet --txt real2clipart.txt --dataset domainnet/clipart \
#  > ./log/main_test_real2clipart.log 2>&1 &

#office-31
 nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
--source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2132-OH_amazon_ce_singe_gpu_resnet50_best.pkl \
--weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240429-2035-OH_amazon_ce_singe_gpu_resnet50_best_neg_param.pth \
--source amazon --target webcam --model_name final_office31 --txt amazon2webcam.txt --dataset office-31/webcam --num_class 31 --select_data Office_31 --num_per_time Office_31 \
 > ./log/office31_main_test_amazon2webcam.log 2>&1 &
nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
--source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2132-OH_amazon_ce_singe_gpu_resnet50_best.pkl \
--weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240429-2035-OH_amazon_ce_singe_gpu_resnet50_best_neg_param.pth \
--source amazon --target dslr --model_name final_office31 --txt amazon2dslr.txt --dataset office-31/dslr --num_class 31 --select_data Office_31 --num_per_time Office_31 \
 > ./log/office31_main_test_amazon2dslr.log 2>&1 &
wait
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2133-OH_dslr_ce_singe_gpu_resnet50_best.pkl \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240428-1923-OH_dslr_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source dslr --target amazon --model_name final_office31 --txt dslr2amazon.txt --dataset office-31/amazon --num_class 31 --select_data Office_31 --num_per_time Office_31 \
#  > ./log/office31_main_test_dslr2amazon.log 2>&1 &
# wait
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2133-OH_dslr_ce_singe_gpu_resnet50_best.pkl \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240428-1923-OH_dslr_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source dslr --target webcam --model_name final_office31 --txt dslr2webcam.txt --dataset office-31/webcam --num_class 31 --select_data Office_31 --num_per_time Office_31 \
#  > ./log/office31_main_test_dslr2webcam.log 2>&1 &
# wait
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2211-OH_webcam_ce_singe_gpu_resnet50_best.pkl \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240428-1956-OH_webcam_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source webcam --target amazon --model_name final_office31 --txt webcam2amazon.txt --dataset office-31/amazon --num_class 31 --select_data Office_31 --num_per_time Office_31 \
#  > ./log/office31_main_test_webcam2amazon.log 2>&1 &
# wait
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/20240427-2211-OH_webcam_ce_singe_gpu_resnet50_best.pkl \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office31/negative/20240428-1956-OH_webcam_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source webcam --target dslr --model_name final_office31 --txt webcam2dslr.txt --dataset office-31/dslr --num_class 31 --select_data Office_31 --num_per_time Office_31 \
#  > ./log/office31_main_test_webcam2dslr.log 2>&1 &


# echo "office31 done"
#domainnet#--data_dir dataset/domainnet/dataset

#  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target clipart --model_name final_domainnet --txt real2clipart.txt --dataset domainnet/clipart --num_class 126 --select_data DomainNet --num_per_time DomainNet \
#  > ./log/domainnet_main_test_real2clipart.log 2>&1 &
# wait
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target painting --model_name final_domainnet --txt real2painting.txt --dataset domainnet/painting --num_class 126 --select_data DomainNet --num_per_time DomainNet \
#  > ./log/domainnet_main_test_real2painting.log 2>&1 &
# wait
# nohup /public/home/wenyb/anaconda3/bin/conda run -n UniDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_real_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_real_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source real --target sketch --model_name final_domainnet --txt real2sketch.txt --dataset domainnet/sketch --num_class 126 --select_data DomainNet --num_per_time DomainNet \
#  > ./log/domainnet_main_test_real2sketch.log 2>&1 &
# wait
# echo "domainnet done"

# #office_home
# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240416-2141-OH_Art_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source Art --target Clipart --model_name final_officehome --txt Art2Clipart.txt --dataset office-home/Clipart --num_class 65 \
#  >> ./log/office_home_main_test_Art2Clipart.log 2>&1 &

# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240416-2141-OH_Art_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source Art --target Product --model_name final_officehome --txt Art2Product.txt --dataset office-home/Product --num_class 65 \
#  >> ./log/office_home_main_test_Art2Product.log 2>&1 &
#  wait

# nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0211-OH_Art_ce_singe_gpu_resnet50_best_param.pth \
# --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240416-2141-OH_Art_ce_singe_gpu_resnet50_best_neg_param.pth \
# --source Art --target RealWorld --model_name final_officehome --txt Art2World.txt --dataset office-home/Real_World --num_class 65 \
#  >> ./log/office_home_main_test_Art2World.log 2>&1 &
# wait
# #  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# # --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240322-1652-OH_Clipart_ce_singe_gpu_resnet50_best_param.pth \
# # --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0004-OH_Clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# # --source Clipart --target Art --model_name final_officehome --txt Clipart2Art.txt --dataset office-home/Art --num_class 65 \
# #  >> ./log/main_test_Clipart2Art.log 2>&1 &
# #  wait

# #  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# # --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240322-1652-OH_Clipart_ce_singe_gpu_resnet50_best_param.pth \
# # --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0004-OH_Clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# # --source Clipart --target Product --model_name final_officehome --txt Clipart2Product.txt --dataset office-home/Product --num_class 65 \
# #  >> ./log/main_test_Clipart2Product.log 2>&1 &

# #  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# # --source_model  /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240322-1652-OH_Clipart_ce_singe_gpu_resnet50_best_param.pth \
# # --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0004-OH_Clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
# # --source Clipart --target RealWorld --model_name final_officehome --txt Clipart2World.txt --dataset office-home/Real_World --num_class 65 \
# #  >> ./log/main_test_Clipart2World.log 2>&1 &

# # wait
# #  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# #  --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0239-OH_Product_ce_singe_gpu_resnet50_best_param.pth \
# #  --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0055-OH_Product_ce_singe_gpu_resnet50_best_neg_param.pth \
# #  --source Product --target Art --model_name final_officehome --txt Product2Art.txt --dataset office-home/Art --num_class 65 \
# #   >> ./log/main_test_Product2Art.log 2>&1 &

# # nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# # --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0239-OH_Product_ce_singe_gpu_resnet50_best_param.pth \
# # --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0055-OH_Product_ce_singe_gpu_resnet50_best_neg_param.pth \
# # --source Product --target Clipart --model_name final_officehome --txt Product2Clipart.txt --dataset office-home/Clipart --num_class 65 \
# #  >> ./log/main_test_Product2Clipart.log 2>&1 &
# #  wait

# # nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# # --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0239-OH_Product_ce_singe_gpu_resnet50_best_param.pth \
# # --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240415-0055-OH_Product_ce_singe_gpu_resnet50_best_neg_param.pth \
# # --source Product --target RealWorld --model_name final_officehome --txt Product2World.txt --dataset office-home/Real_World --num_class 65 \
# #  >> ./log/main_test_Product2World.log 2>&1 &

# #  nohup /public/home/imgbreaker/anaconda3/bin/conda run -n CIUDA --no-capture-output python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
# #  --source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/20240414-0211-OH_RealWorld_ce_singe_gpu_resnet50_best_param.pth \
# #  --weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/office_home/negative/20240416-2141-OH_RealWorld_ce_singe_gpu_resnet50_best_neg_param.pth \
# #  --source RealWorld --target Art --model_name final_officehome --txt World2Art.txt --dataset office-home/Art --num_class 65 \
# #   >> ./log/main_test_World2Art.log 2>&1 &
# #   wait
#   echo "office_home done"


