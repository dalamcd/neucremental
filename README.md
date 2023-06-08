# Overview
Neucremental is a tool for the creation and visualization of neural networks. It provides an interface, network.lua, for creating and training networks, and another interface, visualizer.lua, for displaying that network to the user. It also provides instance.lua, which includes some common tools for each network created.

The visualization componenent uses the [LÖVE 2D framework](https://love2d.org/) for rendering to the screen.

To use, clone this repository and download an OS appropriate version of LOVE from the link above.

### Windows and Mac
Drag and drop the entire cloned repo onto the LOVE executable.

### Linux
Once LOVE is installed, run `love /path/to/neucremental/`

## Basic usage
At startup, the app will look in the networks directory and load any Lua files (it will ignore anything without a .lua extension), then display a list of instances found this way. Once an instance has been selected, the instance will take over and Neucremental will use the load(), update(), and draw() functions--which the instance must implement--to load the instance and render it to the screen.

load() is called once as soon as the instance is selected. Once loaded, on every frame update() is called followed by draw(). Any drawing operations that occur outside of draw() are not valid and won't be rendered.

A few example networks are included 

## Network interface
Creating a new network is simple:
```
network = require("network")
nn = network:new({2, 2, 1})
```
This will create a new network with 2 inputs, one hidden layer with 2 neurons, and 1 output neuron.

Training the network requires an array of arrays with length equal to the number of inputs, and an array of arrays with lengths equal to the number of output neurons. Each parent array must be the same length, and there should be a 1:1 correlation between input indices and expected output indices.

So we might train our example network to operate like an AND gate like this:
```
trainingData = {
  {0, 0},
  {0, 1},
  {1, 0},
  {1, 1},
}

expectedOutputs = {
  {0},
  {0},
  {0},
  {1},
}

nn:learn(trainingData, expectedOutputs)
```
learn() can take an optional third parameter, the learning rate, which defaults to 1.

## Visualization interface and instance library
The instance library provides two helper functions for drawing to the screen, createSurface() and drawSurface(), and most visulization functions take a surface as a parameter. Any visulizations are first drawn to the surface, and then the surfaces are rendered to the screen.
```
instance = require("instance")
visualizer = require("visualizer")

-- this will create a surface at (x, y) position (10, 20), with a width of 200 and a height of 100
networkSurface = instance.createSurface(10, 20, 200, 100)

function draw()
  -- tell LOVE we are drawing to the network surface
  love.graphics.setCanvas(networkSurface.canvas)
    -- use the visualizer to draw the network we created earlier
    visualizer.drawNetwork(networkSurface, nn)
   
  -- tell LOVE we are now drawing to the main canvas
  love.graphics.setCanvas()
  -- render that surface to the screen
  instance.drawSurface(networkSurface)
end
```

### drawGrid() and gridCell()
The visualizer offers two functions which work nicely with createSurface() to divide the screen into a grid and then draw things--e.g. a picture of the network, a graph of the cost, and a view of the latest outputs--to specific sections of that grid.
drawGrid() takes a rows table as a parameter, as well as the total width and height of the grid. Each element of the rows table can be a number or another table. If it is a number N, that row is evenly divided into N columns. If it is a table with M elements, the row is divided into M columns, where the width of each column is a fraction of the sum of the elements. For instance, a row like {3, 1} will be divided into two columns, where the first column takes up 3/4ths of the row and the second column takes up the remaining 1/4th.
Example:
```
rows = {
  3,
  {1, 2, 3},
  2,
}
```
will create a grid like:
```
_____________
|___|___|___|
|__|___|____|
|_____|_____|
```
One important note, drawGrid() expects to be drawn to the main canvas, not a separate surface.

gridCell() takes the same parameters as drawGrid() as well as a row number and column number, and will return the x position, y position, width and height of that cell.

You can use this in conjuction with a createSurface() to easily comparmentalize views. Using the example grid above:
```
rows = { 3, {1, 2, 3}, 2 }
-- create a surface in the cell at row 1, column 1
topLeftSurface = instance.createSurface(visualizer.gridCell(rows, 100, 100, 1, 1)
-- create a surface in the cell at row 2, column 2
middleMiddleSurface = instance.createSurface(visualizer.gridCell(rows, 100, 100, 2, 2)
 -- create a surface in the cell at row 3, column 2
bottomRightSurface = instance.createSurface(visualizer.gridCell(rows, 100, 100, 3, 2)

function draw()
  love.graphics.setCanvas(topLeftsurface)
    visualizer.drawNetwork(topLeftSurface, nn)
    
    ... -- draw to the other surfaces

    -- return to the main canvas
    love.graphics.setCanvas()
    -- draw the grid to the main canvas
    visualizer.drawGrid(rows, 100, 100)
    -- draw the surfaces to the main canvas
    instance.drawSurface(topLeftSurface)
end
```
