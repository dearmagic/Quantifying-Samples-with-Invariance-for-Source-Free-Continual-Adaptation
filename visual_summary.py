import numpy as np
import os

if __name__ == '__main__':
    #read in matrix
    domains = ['Art', 'Clipart', 'Product', 'World']
    cls_num = 65
    dataset_prefix = ''
    # domains = ['amazon', 'dslr', 'webcam']
    # domains = ['clipart', 'painting', 'real', 'sketch']
    # cls_num = 126
    # dataset_prefix = 'domainNet-126_'
    # domains = ['b','c','i','p']
    # cls_num = 12
    # dataset_prefix = 'imageCLEF_'
    for domain in domains:
        #read in matrix
        matrix = np.load('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/{}results/confusion_matrix/{}/matrix_{}_{}.npy'.format(dataset_prefix,domain, domain,domain))

        add_perfomance = np.load('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/{}results/confusion_matrix/{}/add_performance_{}_{}.npy'.format(dataset_prefix,domain, domain,domain))
        mul_perfomance = np.load('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/{}results/confusion_matrix/{}/mul_performance_{}_{}.npy'.format(dataset_prefix,domain, domain,domain))
        #
        matrix_ori = matrix.copy()
        for i in range(cls_num):
            matrix[i,i] = 0
            add_perfomance[i,i] = 0
            mul_perfomance[i,i] = 0

        matrix_reshape = matrix.reshape(-1, 1)
        add_perfomance_reshape = add_perfomance.reshape(-1, 1)
        mul_perfomance_reshape = mul_perfomance.reshape(-1, 1)

        #print the top5 element in matrix and the corresponding cls
        #get index of top 5 element
        
        index = np.argsort(matrix_reshape, axis=0)[-5:]
        print(index)
        print(index%cls_num)
        print(index//cls_num)


        #get the sort matrix
        matrix_sort = np.argsort(matrix_reshape, axis=0)
        add_perfomance_sort = np.argsort(add_perfomance_reshape, axis=0)
        mul_perfomance_sort = np.argsort(mul_perfomance_reshape, axis=0)


        #get the sort element
        matrix_sort_element = matrix_reshape[matrix_sort]
        add_perfomance_sort_element = add_perfomance_reshape[add_perfomance_sort]
        mul_perfomance_sort_element = mul_perfomance_reshape[mul_perfomance_sort]

        #get the top cls_num elements
        matrix_sort_element = matrix_sort_element[-cls_num:]
        add_perfomance_sort_element = add_perfomance_sort_element[-cls_num:]
        mul_perfomance_sort_element = mul_perfomance_sort_element[-cls_num:]


        print(domain)

        #
        most_confusion_cls = []
        most_confusion_cls_add = []
        most_confusion_cls_mul = []
        for i in range(cls_num):
            most_confusion_cls.append((int(matrix_sort[i-cls_num:][0])%cls_num, int(matrix_sort[i-cls_num:][0])//cls_num))
            most_confusion_cls_add.append((int(add_perfomance_sort[i-cls_num:][0])%cls_num, int(add_perfomance_sort[i-cls_num:][0])//cls_num))
            most_confusion_cls_mul.append((int(mul_perfomance_sort[i-cls_num:][0])%cls_num, int(mul_perfomance_sort[i-cls_num:][0])//cls_num))

        import matplotlib.pyplot as plt

        most_confusion_cls_matrix = np.zeros((cls_num, cls_num))
        most_confusion_cls_add_matrix = np.zeros((cls_num, cls_num))
        most_confusion_cls_mul_matrix = np.zeros((cls_num, cls_num))

        for i in range(cls_num):
            most_confusion_cls_matrix[most_confusion_cls[i][0], most_confusion_cls[i][1]] = 1
            most_confusion_cls_add_matrix[most_confusion_cls_add[i][0], most_confusion_cls_add[i][1]] = 1
            most_confusion_cls_mul_matrix[most_confusion_cls_mul[i][0], most_confusion_cls_mul[i][1]] = 1
        save_path = '/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/{}results/confusion_matrix/{}/'.format(dataset_prefix,domain)
        np.save(save_path + 'most_confusion_cls.npy', most_confusion_cls)
        np.save(save_path + 'most_confusion_cls_add.npy', most_confusion_cls_add)
        np.save(save_path + 'most_confusion_cls_mul.npy', most_confusion_cls_mul)
        # fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        # ax[0].imshow(most_confusion_cls_matrix, cmap='hot', interpolation='nearest')
        # ax[0].set_title('most confusion cls matrix')
        # ax[1].imshow(most_confusion_cls_add_matrix, cmap='hot', interpolation='nearest')
        # ax[1].set_title('most confusion cls add matrix')
        # ax[2].imshow(most_confusion_cls_mul_matrix, cmap='hot', interpolation='nearest')
        # ax[2].set_title('most confusion cls mul matrix')
        log_log_matrix = np.log(matrix_ori+1)+1

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(log_log_matrix, interpolation='nearest')
        # ax.set_title('log log matrix')

        # ax[1].imshow(add_perfomance, cmap='hot', interpolation='nearest')
        # ax[1].set_title('most confusion cls add matrix')
        # ax[2].imshow(mul_perfomance, cmap='hot', interpolation='nearest')
        # ax[2].set_title('most confusion cls mul matrix')
        #set font size
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        if not os.path.exists('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/results/confusion_matrix/{}'.format(domain)):
            os.makedirs('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/results/confusion_matrix/{}'.format(domain))
        #remove axis
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig('/public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/results/confusion_matrix/{}/log_log_matrix.png'.format(domain))

        plt.close()
        print('done')