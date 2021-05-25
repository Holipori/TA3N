import os



def modify_list(file_path, new_path):
    with open(file_path) as f:
        fo = open(new_path, "w")
        for line in f.readlines():
            content = line.split(' ')
            frames = os.listdir('/home/xinyue/'+ content[0])
            # print(frames)
            frame_num = len(frames)
            # if 'Goal_1__2_kick_ball_f_cm_np1_fr_goo_2' in content[0]:
            #     print(content)
            #     new_content = content[0] + ' ' + str(frame_num) + ' ' + content[2]
            #     print(new_content)
            #     break
            content[0] = content[0].replace('RGB-feature', 'RGB-feature-i3d')
            new_content = content[0] +' '+ str(frame_num) +' '+ content[2]
            print(new_content)
            fo.write(new_content)
            # print(frame_num)
            # print(content)
            # break
        fo.close()

if __name__ == '__main__':
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf-feature.txt'
    new_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf-feature.txt'
    new_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_hmdb_ucf-feature.txt'
    new_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_hmdb_ucf-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_hmdb_ucf-feature.txt'
    new_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_hmdb_ucf-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_hmdb_ucf_small-feature.txt'
    new_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_hmdb_ucf_small-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_hmdb_ucf_small-feature.txt'
    new_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_hmdb_ucf_small-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf_small-feature.txt'
    new_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf_small-feature-i3d.txt'
    modify_list(file_path, new_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf_small-feature.txt'
    new_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf_small-feature-i3d.txt'
    modify_list(file_path, new_path)