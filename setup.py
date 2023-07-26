#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='decode_lab_code',  
     version='0.1',
     scripts=[''] ,
     author="John Stout",
     author_email="john.j.stout.jr@gmail.com",
     description="A python package to support data analysis in the DECODE lab @ Nemours Childrens Hospital in Wilmington DE, led by A.Hernan and R.Scott",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/javatechy/dokr",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BDS3 License",
         "Operating System :: OS Independent",
     ],
 )