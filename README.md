This is an impementation of the paperby Gatys et. al on Neural style. It generates a stylized image from a photo, by applying the style of the style image.
It takes 2 images , present in the same folder as the current file.
One is the content image and the other the style image.  The style image is resized to match the content image before we generate a new image
The algorithm aims to minizime overall loss, which is the sum of style loss and content loss.
Style and content losses are not calculated against the original images ( not a per pixel matching approach). Instead, the images are run through the VG16 network ( class attached in same folder). The output of the VGG network is what is looked at when minimizing the error.
One can control the relative balance of presence of original photographic content vs style .  alpha -> controls fraction of content loss to consider, beta-> fraction of style loss.  Gatys suggest a 1:1000 ratio, where the style loss is scaled 1000 times relative to the content. 
Avoid using large images, as they will proceed more slowly. The GPU RAM is a limitation on the size of the images you can process. 
For generating the output images, one can start with a blank white noise image, or with the original content image. For portraits, I found the content image as a better starting point as it preserved fine details near the eyes. 
