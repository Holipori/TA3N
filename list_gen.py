import os



def modify_list(file_path, new_path):
    with open(file_path) as f:
        fo = open(new_path, "w")
        for line in f.readlines():
            content = line.split(' ')
            content[0] = content[0].replace('RGB-feature', 'RGB-feature-i3d')
            frames = os.listdir('/home/xinyue/'+ content[0])
            # print(frames)
            frame_num = len(frames)
            # if 'Goal_1__2_kick_ball_f_cm_np1_fr_goo_2' in content[0]:
            #     print(content)
            #     new_content = content[0] + ' ' + str(frame_num) + ' ' + content[2]
            #     print(new_content)
            #     break
            new_content = content[0] +' '+ str(frame_num) +' '+ content[2]
            print(new_content)
            fo.write(new_content)
            # print(frame_num)
            # print(content)
            # break
        fo.close()

def i3d_to_flow(file_path):
    new_path = file_path.replace('i3d.txt', 'flow.txt')
    with open(file_path) as f:
        fo = open(new_path, "w")
        for line in f.readlines():
            content = line.split(' ')
            content[0] = content[0].replace('RGB-feature-i3d', 'RGB-feature-flow')
            frames = os.listdir('/home/xinyue/' + content[0])
            frame_num = len(frames)
            new_content = content[0] + ' ' + str(frame_num) + ' ' + content[2]
            print(new_content)
            fo.write(new_content)
            # print(frame_num)
            # print(content)
            # break
        fo.close()

if __name__ == '__main__':
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf-feature-i3d.txt'
    i3d_to_flow(file_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_train_hmdb_ucf_small-feature-i3d.txt'
    i3d_to_flow(file_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf-feature-i3d.txt'
    i3d_to_flow(file_path)
    file_path = '/home/xinyue/dataset/hmdb51/list_hmdb51_val_hmdb_ucf_small-feature-i3d.txt'
    i3d_to_flow(file_path)

    # file_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)
    # file_path = '/home/xinyue/dataset/olympic/list_olympic_train_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/olympic/list_olympic_train_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)
    # file_path = '/home/xinyue/dataset/olympic/list_olympic_val_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/olympic/list_olympic_val_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)
    # file_path = '/home/xinyue/dataset/olympic/list_olympic_train_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/olympic/list_olympic_train_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)
    # file_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/ucf101/list_ucf101_train_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)
    # file_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_ucf_olympic-feature.txt'
    # new_path = '/home/xinyue/dataset/ucf101/list_ucf101_val_ucf_olympic-feature-i3d.txt'
    # modify_list(file_path, new_path)