# Plan

## Goal

An image containing only dead sperms and big dust.



1. We need to remove small dust.
2. We need to remove big spermy patches.


## Plan to get to only big dust and sperm level

Let us say we have 4 things

1. dead sperms (DS)
2. big dust (BD)
3. alive sperms (AS)
4. small dust (SD)

## Plan 

1. Now we have BD+DS+AS+SD.
2. We will first do hue separation. output is AS + (half)DS. 
3. We will invert above image and use it as mask. output is BD + SD + (half)DS
4. We will do erosion on above image and get BD + (half)DS
5. Now we will merge(bitwise_or) with AS + (half)DS and we will get BD + DS + AS.

Above steps are done and now we have a beautiful image with no dirt. (dirtless.jpg)

## Next Plan

1. Now we are going to apply cca to the dirtless image and form bounding boxes.
2. Next we crop the images and store them all in a single folder. 
3. Labelling of images comes next
4. Set up classifier 
5. predict







