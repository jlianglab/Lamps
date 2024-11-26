import os
import cv2

imgpath = '/mnt/sda/zhouziyu/ssl/datasets/fundus_datasets/AIROGS'

# file_train = open('./data/fundus/RFMiD2.0/train.txt', 'w')
# file_test = open('./data/fundus/RFMiD2.0/test.txt', 'w')
# file_val = open('./data/fundus/RFMiD2.0/val.txt', 'w')
# for i in os.listdir(os.path.join(imgpath,'Training_Set/Training')):
#     file_train.writelines('Training_Set/Training/'+i+'\n')
# file_train.close()

# for i in os.listdir(os.path.join(imgpath,'Test_Set/Test')):
#     file_test.writelines('Test_Set/Test/'+i+'\n')
# file_test.close()

# for i in os.listdir(os.path.join(imgpath,'Evaluation_Set/Validation')):
#     file_val.writelines('Evaluation_Set/Validation/'+i+'\n')
# file_val.close()

# file_train = open('./data/fundus/PALM/test.txt', 'w')

# for i in os.listdir(os.path.join(imgpath,'ODIR-5K_Testing_Images')):
#     file_train.writelines('ODIR-5K_Testing_Images/'+i+'\n')
# file_train.close()

file_train = open('./data/fundus/AIROGS/train.txt', 'w')
for i in range(6):
    for j in os.listdir(os.path.join(imgpath, str(i))):
        file_train.writelines(f'{i}/'+j+'\n')
file_train.close()

