import os

def main():
    labels = ['dust', 'degraded', 'big_halo', 'small_halo', 'medium_halo']
    for label in labels:
        print("label len",label, len(os.listdir(label)))

if __name__ == '__main__':
    main()
