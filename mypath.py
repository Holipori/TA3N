class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/home/xinyue/dataset/ucf101/RGB'

            # Save preprocess data into output_dir
            output_dir = '/home/xinyue/dataset/ucf101/out2'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = '/home/xinyue/dataset/hmdb51/RGB'

            output_dir = '/home/xinyue/dataset/hmdb51/out2'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return './c3d-pretrained.pth'