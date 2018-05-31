This reprository contains the code I made to make a bumblebee detector using green screen technology. I goes with the blog post https://medium.com/@jannekool/the-fabrication-of-a-simple-bumblebee-detector-using-green-screen-technology-e7665da9874a The text of the blog is below, for images and motion pictures you need to click the link.



The fabrication of a simple bumblebee detector using green screen technology

Following the recent news about nature, one could get overwhelmed by the bad news about insects decline in general and bees in particular. To get a better understanding of the role of insects in the world and to be able to monitor their wellbeing it would be great to have tools that enable automatic data collection. In this blog post I describe how I made a bumblebee detector using a green screen like your local weather man uses.



In a previous post How to use deep learning to quantify pollinator behavior I described how I filmed Dahlia flowers and used transfer-learning to find the frames on which there was a bumblebee present. This way I could easily count how many bumblebees actually visited the flower (20 in 90 minutes). Unfortunately, the network did not generalize to other flowers so I decided to create new data with plenty of bumblebees with all kinds of lighting and backgrounds. To do so I needed all kind pictures of bumblebees with different lightning. It was no option to simply download images of bumblebees from the internet. These tend to be very nice and artistic, and the bumblebee detector I built on this data indeed recognized bumblebees from images downloaded from the internet, but did not find the bumblebees on my not so sharp crappy images. So I wanted to collect diverse images with the same web-cam I use for data collection. Also it was winter making it a bit hard to find bumblebees outside.
So to collect images I built a tent in my living room, planted flowers in it and covered the bottom with some green paper. Next, I ordered a bumblebeehive on-line. The bumblebees where delivered to our small apartment, and during daytime they nicely flew around, making a quite incredible noise, a lot more than a vacuum cleaner. Fortunately they sleep at night, and during the experiment the night was long. Here is a picture of the setting:
They I actually did not fly that nicely, they were clearly more interested in getting out of the tent then in the nice flowers and sugar water presented. However, the smarter ones got in front of the green screen and the images I made of those resulted easily in a thousand pictures of bumblebees. I wrote some code to cut them out from the green background, which you can find on my github account.
From the internet I downloaded a thousand pictures of flowers, more specifically daisies and dandelions, I might include more varied backgrounds later. Next I wanted to glue the images of bees on the flower images automatically such that the lightning of the bumblebee would be similar to the lightning of the image. To do so, I made an affine linear transformation that scaled the bee such that the vectors representing each pixel in the array was about the same size as the the vectors in the background image around the place where I wanted to glue it in. Also, I wanted to have the boundary of the bee a little bit smoothed out. So I glued in the bee following for each pixel the formula
new pixel = (1-weight) * oldpixel+weight* beepixel,
where the weight was zero at the boundary and stepwise growing to 1 in the middle of the bumblebee. Again, the code I used can be found on github, here are some images I created:



To make a simple detector I made a convolutional neural network in Keras on cuts from the flowers of 45 x 45 x 3 arrays, containing a bee or not. I tried to keep the network small but still with good performance. Afterwards, I used a simple slide and detect algorithm to find the bumblebees. That is, I slided my viewing window over the image to get a probability that there was a bumblebee. Often when a bee is there, two locations close to each other both get a high probability. I decided to filter for that, which has as a drawback that if two bees come close to each other I will often loose track of one of those. However, in more natural situations than my living room, bumblebees are not likely to be seen crawling over each other. In my living room they did, as you can see in the following motion picture:



So its not perfect but it actually works quite nice. All detections are indeed bees, but quite some are overlooked. The bees in other datasets can be much larger or smaller then 45 x 45, so I am working on picking the right solution for that. Moreover, the slide and check algorithm is simple indeed, but computationally expensive.
Right now summer is present again and the bumblebees are alive and kicking in the field. The coming months I will be collecting more outdoor data to test and refine the green screen method and some other methods and ideas. One thing one could easily do is fly some other insects in similar green screen backgrounds and add more species to the classifier.

Finally I would like to thank the DTU-compute-cogsys group at the Danish technical university for their warm welcome and sharing of ideas and facilities.
