import augly.image as imaugs
import os
import random
import shutil

random.seed(0)


def rotate(img_path):
    random_degree = random.randrange(45, 305)
    head, tail = os.path.split(img_path)
    new_img_path = os.path.join(head, f'rotated_{random_degree}___' + tail)
    imaugs.rotate(image=img_path, output_path=new_img_path, degrees=random_degree)

def saturate(img_path):
    random_saturation_factor = 1.5 - random.random()
    head, tail = os.path.split(img_path)
    new_img_path = os.path.join(head, f'saturated_{random_saturation_factor}___' + tail)
    imaugs.saturation(image=img_path, output_path=new_img_path, factor=random_saturation_factor)

def add_noise(img_path):
    noise_mean = 0.5*random.random()
    head, tail = os.path.split(img_path)
    new_img_path = os.path.join(head, f'noise_{noise_mean}_0___' + tail)
    imaugs.random_noise(image=img_path, output_path=new_img_path, mean=noise_mean, seed=0)

def brighten(img_path):
    random_brightness_factor = 1.5 - random.random()
    head, tail = os.path.split(img_path)
    new_img_path = os.path.join(head, f'brightness_{random_brightness_factor}___' + tail)
    imaugs.brightness(image=img_path, output_path=new_img_path, factor=random_brightness_factor)

def identity(_):
    pass

def factor_4_point_5_aug(img_path):
    task_list = [
        saturate,
        rotate,
        rotate,
        add_noise,
    ]
    random.shuffle(task_list)
    if random.random() < 0.5:
        task_list = task_list[:-1]
    task_list += [identity]
    for task in task_list:
        task(img_path)


def factor_1_point_5_aug(img_path):
    task_list = [
        saturate,
        rotate,
        add_noise,
    ]
    random.shuffle(task_list)
    task_list = task_list[:-2]
    if random.random() < 0.5:
        task_list = task_list[:-1]
    task_list += [identity]
    for task in task_list:
        task(img_path)

def factor_7_aug(img_path):
    task_list = [
        saturate,
        saturate,
        rotate,
        rotate,
        add_noise,
        add_noise,
        identity,
    ]
    for task in task_list:
        task(img_path)

def sample_only(img_path, n):
    head, tail = os.path.split(img_path)
    total_images = len(os.listdir(head))
    if n >= total_images:
        return
    discarded_folder = os.path.join(head, 'discarded')
    if not os.path.isdir(discarded_folder):
        try:
            os.mkdir(discarded_folder)
        except Exception as e:
            raise e
    
    
    if random.random() > (n/total_images):
        src_path = img_path
        dst_path = os.path.join(discarded_folder, tail)
        shutil.move(src_path, dst_path)


transform_func = {
    'dust': (sample_only, [3000]),
    'degraded': (factor_4_point_5_aug, []),
    'big_halo': (factor_7_aug, []),
    'small_halo': (factor_1_point_5_aug, []),
    'medium_halo': (factor_4_point_5_aug, [])
}

## run this from inside labelled_sorted
def main():
    here_path = os.getcwd()
    print("here", here_path)
    for dir in os.listdir(here_path):
        image_dir = os.path.join(here_path, dir)
        print("here", dir, image_dir)
        if not os.path.isdir(image_dir):
            continue
        print("dir", dir, image_dir)
        for image_file in os.listdir(image_dir):
            image_file_path = os.path.join(image_dir, image_file)
            if not os.path.isfile(image_file_path):
                continue
            print("image_file", image_file, image_file_path)
            transform_func[dir][0](image_file_path, *transform_func[dir][1])    
    

if __name__ == '__main__':
    main()
