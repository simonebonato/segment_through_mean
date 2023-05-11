# Longest Segment Through Mean

This is a simple method that, given a list of 2d points that constitutes the border of a shape, finds all the segments that start from the points on the border, pass through the mean of the points, and end on the border again. Then it finds the longest segment among them.

## :snake: How to use 
First you need to have (or create) an environment using the `requirements.txt` file.
To use the code, simply clone the repo and use the code contained in the `extractor_class.py` file, as shown in `example.ipynb`. 

The code is really simple, don't get discouraged by the length of the code on the notebook, most of it belongs to the plotting part!

 ## :whale2: Example figures 
An example of the figure used in the example notebook is:
![image](https://github.com/simonebonato/segment_through_mean/assets/63954877/d706bdfe-a750-4233-b3d9-fc1a7877f068)

Then it is possible to extract all the edges that make the figure:
![image](https://github.com/simonebonato/segment_through_mean/assets/63954877/250de978-62b2-454d-a318-9fdc9573902b)

And finally compute all the segments that start from the points that define the border and arrive on the other side, finding the longest one:
![image](https://github.com/simonebonato/segment_through_mean/assets/63954877/d4eca877-234d-457f-b438-cb6accdd1730)