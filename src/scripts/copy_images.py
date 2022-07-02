import shutil, os

label_list = [
    'degraded',
    'dust',
    'big_halo',
    'medium_halo',
    'small_halo'
]
CURRPATH = os.getcwd()


def create_new_folders():
    
    for label in label_list:
        try:
            os.makedirs(os.path.join(CURRPATH, '..','labelled_sorted', label))
        except FileExistsError as fee:
            pass
        except Exception as e:
            raise Exception('dir not created', e)


def copy_image(src_path, new_folder_path):
    _, tail = os.path.split(src_path)
    dst_path = os.path.join(new_folder_path, tail)
    shutil.copy(src_path, dst_path)

def walkdi():
    for root, dirs, files in os.walk(CURRPATH):
        for name in files:
            full_file_path = os.path.join(root, name)
            _, ext = os.path.splitext(full_file_path)
            if ext != '.jpg' or os.path.isdir(full_file_path):
                continue
            # print("start", full_file_path)
            if 'dust' in full_file_path:
                new_folder_path = os.path.join(CURRPATH,'..','labelled_sorted', 'dust')
            elif 'nondegraded' in full_file_path:
                print("ffp", full_file_path)
                if 'big' in full_file_path:
                    new_folder_path =  os.path.join(CURRPATH,'..','labelled_sorted', 'big_halo')
                elif 'small' in full_file_path:
                    new_folder_path =  os.path.join(CURRPATH,'..','labelled_sorted', 'small_halo')
                if 'medium' in full_file_path:
                    new_folder_path =  os.path.join(CURRPATH,'..','labelled_sorted', 'medium_halo')
            elif 'degraded' in full_file_path:
                new_folder_path = os.path.join(CURRPATH,'..','labelled_sorted', 'degraded')
            else:
                continue
            copy_image(full_file_path, new_folder_path)

        # for name in dirs:
        #     print(os.path.join(root, name))
## run it from inside labelled
def main():
    create_new_folders()
    walkdi()


if __name__ == '__main__':
    main()