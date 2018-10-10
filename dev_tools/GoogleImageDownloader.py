#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:39:28 2018

@author: pbower
"""

from google_images_download import google_images_download


# Download 100 pictures of Milhouse
response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {"keywords":"milhouse -'milhouse van houten'","prefix":"pic_", "color_type":"full-color", "format":"jpg", "offset":101}    #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images



# Download 100+ pictures of Principal Skinner
response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {"keywords":"Seymour Skinner Principal Simpsons","limit":100, "prefix":"pic_", "color_type":"full-color", "format":"jpg"}    #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images





# Download 100 pictures of Krusty
response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {"keywords":"Krusty the Clown","limit":100, "prefix":"pic_", "color_type":"full-color", "format":"jpg"}    #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images





# Download 50 pictures of Mr Burns
response = google_images_download.googleimagesdownload()   # class instantiation

arguments = {"keywords":"Mr Burns","limit":50, "prefix":"pic_", "color_type":"full-color", "format":"jpg"}    #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images




