import numpy as np
import os
from PIL import Image
from torchvision import transforms
from scipy.interpolate import interp1d

def constrcut_negative_dataset(classA, classB):
    #mix the picture from classA and classB
    #return the mixed dataset
    #randomly select the same number of picture from classA and classB
    pass


if __name__ == '__main__':
    #load the most confusion class
    # domains = ['Art', 'Clipart', 'Product', 'World']
    # domains = ['amazon', 'dslr', 'webcam']
    # load_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/office31_results/confusion_matrix/'
    # load_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/domainNet-126_results/confusion_matrix/'
    load_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/imageCLEF_results/confusion_matrix/'
    domains = ['b','c','i','p']

    # get the name and index of each class
    # file_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/dataset/office-home/Art.txt'
    # file_path = '/public/home/imgbreaker/Desktop/PADA/pytorch/data/office/amazon_31_list.txt'
    # file_path = '/public/home/imgbreaker/Desktop/dataset/DomainNet/data/clipart_list.txt'
    file_path = '/public/home/imgbreaker/Desktop/PADA/pytorch/imageCLEF/list/bList.txt'
    original_path = '/public/home/imgbreaker/Desktop/PADA/pytorch/imageCLEF/'
    print('finish init')
    with open(file_path, 'r') as f:
        lines = f.readlines()
    class_dict = {}
    for line in lines:
        line = line.strip()
        class_name = line.split(' ')[0].split('/')[-2]
        class_index = line.split(' ')[1]
        if class_index not in class_dict.keys():
            class_dict[class_index] = class_name
    transform_ = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
    ])
    print('finish load the class dict')

    for domain in domains:
        confusion_pair = np.load(load_path + domain + '/most_confusion_cls_add.npy')
        #find all the index of which is not zero
        for index in confusion_pair:
            classA = class_dict[str(index[0])]
            classB = class_dict[str(index[1])]
            #randomly readin the picture from classA and classB randomly and mix them together (segement by 四次曲线)
            # pathA = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/dataset/office-home/'+ domain + '/' + classA + '/'
            # pathB = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/dataset/office-home/'+ domain + '/' + classB + '/'
            # pathA = '/public/home/imgbreaker/Desktop/PADA/pytorch/dataset/office/domain_adaptation_images/' + domain + '/images/' + classA + '/'
            # pathB = '/public/home/imgbreaker/Desktop/PADA/pytorch/dataset/office/domain_adaptation_images/' + domain + '/images/' + classB + '/'
            pathA = original_path + domain + '/' + classA + '/'
            pathB = original_path + domain + '/' + classB + '/'

            #get the name of the picture
            picsA = os.listdir(pathA)
            picsB = os.listdir(pathB)

            num_pics = min(len(picsA), len(picsB))

            selectdA = []
            selectdB = []
            #mix the picture from classA and classB
            pic_name = 0
            for i in range(num_pics):
                #randomly select the picture from classA and classB which is not been selected
                indexA = np.random.randint(0, len(picsA))
                indexB = np.random.randint(0, len(picsB))
                while True:
                    if indexA not in selectdA and indexB not in selectdB:
                        selectdA.append(indexA)
                        selectdB.append(indexB)
                        break
                    indexA = np.random.randint(0, len(picsA))
                    indexB = np.random.randint(0, len(picsB))
                picA = picsA[indexA]
                picB = picsB[indexB]
                #read the picture
                imgA = Image.open(pathA + picA)
                imgB = Image.open(pathB + picB)
                #mix the picture
                imgA = transform_(imgA)
                imgB = transform_(imgB)
                # print(imgA)

                mask = np.zeros(imgA.shape)

                HMAX, WMAX = imgA.shape[1], imgA.shape[2]
                window = 20
                center = (HMAX / 2, WMAX / 2)

                rand_center_point = (np.random.randint(low=int(center[0] - window), high=int(center[0] + window)),
                                     np.random.randint(low=int(center[1] - (window / 3)),
                                                       high=int(center[1] + (window / 3))))

                points = [(0, np.random.randint(HMAX)), rand_center_point, (WMAX - 1, np.random.randint(HMAX))]

                px = [p[0] for p in points]
                px.extend([0, 223])

                py = [p[1] for p in points]
                py.extend([0, 223])

                # spl = spline_interpolation([p[0] for p in points], [p[1] for p in points], range(num_points), order=7, kind='smoothest')
                spl = interp1d([p[0] for p in points], [p[1] for p in points], kind='quadratic')(range(imgA.shape[2]))

                for j in range(imgA.shape[1]):
                    for i in range(imgA.shape[2]):
                        if i < spl[j]:
                            mask[:, j, i] = 1
                        else:
                            mask[:, j, i] = 0


                #generate the mask

                flag_main = np.random.randint(0, 2)

                if flag_main == 0:
                    img = np.multiply(imgA,mask) + np.multiply(imgB, 1 - mask)
                else:
                    img = np.multiply(imgB, mask) + np.multiply(imgA, 1 - mask)
                # input(img)
                #save the picture
                # save_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/dataset/negative_dataset/'+ domain + '/' + classA + '_' + classB + '/'
                save_path = original_path+ 'negative_dataset/'+ domain + '/' + classA + '_And_' + classB + '/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                name_str = str(pic_name)
                name_str = name_str.zfill(5)
                save_name = name_str + '.jpg'
                pic_name += 1
                img = img.numpy()
                img = img*255
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(np.uint8(img))
                img.save(save_path + save_name)
                #img show
            print('finish the mix of classA and classB:', classA, classB)



