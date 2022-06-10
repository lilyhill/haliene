# haliene
Ai for detecting dna fragmentation index

## to redo experiments

just run `python whatever.py`

1. hue_separation2.py gave some really good result - only_living.png
2. cca1.py or connected component analysis was not that successful due to sparsity of pixels in patches. 
3. inpainting1.py is giving a better image but it is still not dense enough for good results on CCA. Notice I have used 20 as pixel radius. update: it was dense enough. Read point 6 
4. Next attempt is to run inpainting1.py in loop, improving with every iteration. Update: it did not work and created a really bad image. (only_living_inpainted2.jpg)
5. Next I am trying to blur the image. update: blurring did not work.
6. Now cca works. I had got it wrong. I was using binary_inv filter for thresholding in cca (what a big mistake). but even after binary filter it had many small dots. I just applied a min cap on area and boom it works!! I used only_living_inpainted.jpg as the input image.


## For noise reduction

1. in erosion.py, we first cleared all the living ones and it leaves black holes for living ones. We tried filling it up with inpaint but result is still a blur. (without_living.jpg) 



## REAL TODOs

1. We are currently creating a lot of image objects and we don't need them. Ideally, there should only be one original image and other image that just changes state and when it changes state it modifies its own frame. We should do it. 

I did it but left a very bad code behind. Please fix it. description is there in main.py

Also should I create new object for image or should I change and return it?