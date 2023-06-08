-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- A simple network which operates like a[n] XOR gate. Expected
-- outputs for AND, NAND and OR gates are included as well, which
-- you can use in place of xorExpected in update() to test other
-- gates.
-- One interesting example, a network with only 2 inputs and no
-- hidden layers can successfully work like AND/NAND/OR, but will
-- be unable to find a solution for XOR. The provided network--with 
-- two inputs and a single hidden layer with two neurons--is the
-- simplest network I know of that can find a solution for XOR.
-- ==============================================================

local v = require("visualizer")
local network = require("network")
local instance = require("instance")

-- the set of possible input states for a gate with 2 inputs (one state per line)
local inputs = {
	{0, 0},
	{0, 1},
	{1, 0},
	{1, 1},
}
-- the correct outputs for a gate given the inputs above
local xorExpected = {
	{0},
	{1},
	{1},
	{0},
}
local andExpected = {
	{0},
	{0},
	{0},
	{1},
}
local orExpected = {
	{0},
	{1},
	{1},
	{1},
}
local nandExpected = {
	{1},
	{1},
	{1},
	{0},
}

local sw, sh = love.graphics.getWidth(), love.graphics.getHeight()

local nn
local grid
local networkSurface
local outputSurface1
local outputSurface2
local outputSurface3
local outputSurface4

local function load()
	-- initialize a network with 2 inputs, 1 hidden layer with 2 neurons, and 1 output
	nn = network:new({2, 2, 1})
	-- a table representing the grid where we will draw our network
	-- see the documentation on renderGrid()
	grid = {1, 4}
	-- create drawing surfaces to display the network and our test outputs
	networkSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 1))
	outputSurface1 = instance.createSurface(v.gridCell(grid, sw, sh, 2, 1))
	outputSurface2 = instance.createSurface(v.gridCell(grid, sw, sh, 2, 2))
	outputSurface3 = instance.createSurface(v.gridCell(grid, sw, sh, 2, 3))
	outputSurface4 = instance.createSurface(v.gridCell(grid, sw, sh, 2, 4))
end

local function update(dt)
	-- train the network
	nn:learn(inputs, xorExpected)
end

local function draw()

	-- draw the grid
	v.renderGrid(grid, sw, sh)

	-- update the surfaces

	-- switch to the canvas for that surface
	love.graphics.setCanvas(outputSurface1.canvas)
		-- send the first of the possible inputs through the network
		nn:forward(inputs[1])
		-- paint the canvas with the state of the network after receiving the input
		v.drawOutputs(outputSurface1, nn)
		-- drawOutputs() doesn't indicate anything about the input, so we'll print
		-- what input this represents
		love.graphics.print("(0, 0)", 70, 140)

	-- repeat for the other possible inputs
	love.graphics.setCanvas(outputSurface2.canvas)
		nn:forward(inputs[2])
		v.drawOutputs(outputSurface2, nn)
		love.graphics.print("(0, 1)", 70, 140)

	love.graphics.setCanvas(outputSurface3.canvas)
		nn:forward(inputs[3])
		v.drawOutputs(outputSurface3, nn)
		love.graphics.print("(1, 0)", 70, 140)

	love.graphics.setCanvas(outputSurface4.canvas)
		nn:forward(inputs[4])
		v.drawOutputs(outputSurface4, nn)
		love.graphics.print("(1, 1)", 70, 140)

	-- switch to the network canvas
	love.graphics.setCanvas(networkSurface.canvas)
		-- paint to the network canvas
		v.drawNetwork(networkSurface, nn, true)

	-- switch back to the main canvas
	love.graphics.setCanvas()
	-- draw the surfaces onto the main canvas
	instance.renderSurface(networkSurface)
	instance.renderSurface(outputSurface1)
	instance.renderSurface(outputSurface2)
	instance.renderSurface(outputSurface3)
	instance.renderSurface(outputSurface4)
end

return {
	label = "XOR gate",
	load = load,
	update = update,
	draw = draw,
}