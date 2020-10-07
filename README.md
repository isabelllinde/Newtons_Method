# Newtons Method
Group project for Object Oriented Programming with Applications course taken at the University of Edinburgh for MSc in Computational Applied Mathematics.

The C# program uses Newton's method to establish a relationship between y<sub>t</sub> (population in millions) and t (periods of time) in the population of the UK (in millions) between 1955 and 2015 csv. The program tests out three different models:
1. y<sub>t</sub> = x<sub>1</sub>e<sup>x<sub>2</sub>t</sup>
2. y<sub>t</sub> = x<sub>1</sub> / (1+x<sub>2</sub>e<sup>x<sub>3</sub>t</sup>)
3. y<sub>t</sub> = x<sub>1</sub> + x<sub>2</sub>t + x<sub>3</sub>t<sup>2</sup> + x<sub>4</sub>t<sup>3</sup>

The program also implements two clustering methods, the greedy approach and the fixed number of clusters approach, and applies them for Euclidean distance, Manhattan distance and Mahalanobis distance across the data in files cluster2.csv and cluster4.csv. The methods allocate the appropriate number of clusters for each dataset. 

# Contributors
Isabell Linde<br/>
Scott Dallas<br/>
