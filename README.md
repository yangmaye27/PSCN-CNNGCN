this dir for the main training and test, you can train by your self after unzip our provided data or just load
our model to test performance
training code example: python train.py --gpu_list 0 1 2 3 --add_note T1 (T1/T2/T3 to select the model type)
evaluation code example: python evaluation.py --gpu_list 0 1 2 3 --add_note T3_test --gpath <path to model G> --dpath <path to model D>

python evaluation.py --gpu_list 0 --add_note T1_test --gpath .//models//modelG_epoch49_batchsize2.pth --dpath ./models/modelD_epoch49_batchsize2.pth

python train.py --gpu_list 0 --add_note T2

python evaluation.py --gpu_list 0 --add_note T2_test --gpath .//models//modelG_epoch49_batchsize16.pth --dpath ./models/modelD_epoch49_batchsize16.pth

model3第一次（随机大小）
python evaluation.py --gpu_list 0 --add_note T3_test --gpath .//models//mode3lG_epoch48_batchsize8.pth --dpath ./models/mode3lD_epoch48_batchsize8.pth

model第二次（固定大小120pix）
python evaluation.py --gpu_list 0 --add_note T3_test --gpath .//models//modelG120_epoch49_batchsize8.pth --dpath ./models/modelD120_epoch49_batchsize8.pth

测试三个模型:
python evaluation.py --gpu_list 0 --add_note T1_test --gpath ./models/Model1Gen.pth --dpath ./models/Model1Dis.pth
python evaluation.py --gpu_list 0 --add_note T2_test --gpath ./models/Model2Gen.pth --dpath ./models/Model2Dis.pth
python evaluation.py --gpu_list 0 --add_note T3_test --gpath ./models/Model3Gen.pth --dpath ./models/Model3Dis.pth

python evaluation.py --gpu_list 0 --add_note T2_test --gpath ./models/2-modelG.pth --dpath ./models/2-modelD.pth







python evaluation.py --gpu_list 0 --add_note T3_test --gpath ./models/mode3lG_epoch49_batchsize8.pth --dpath ./models/mode3lD_epoch49_batchsize8.pth	

python evaluation.py --gpu_list 0 --add_note T3_test --gpath ./models/modelG_epoch50_batchsize8.pth --dpath ./models/modelD_epoch50_batchsize8.pth

python evaluation.py --gpu_list 0 --add_note T1_test --gpath ./models/Model1Gen.pth --dpath ./models/Model1Dis.pth



python evaluation.py --gpu_list 0 --add_note T2_test --gpath ./models/Model2Gen.pth --dpath ./models/Model2Dis.pth

python evaluation.py --gpu_list 0 --add_note T3_test --gpath ./models/Model3Gen.pth --dpath ./models/Model3Dis.pth

python evaluation.py --gpu_list 0 --add_note T3_test --gpath ./models/Model3GenGraph.pth --dpath ./models/Model3DisGraph.pth