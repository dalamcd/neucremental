-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- A network which takes as input the x,y coordinates of an image
-- and outputs the rgb values for that pixel
-- The network is trained on a 28x28 image with the pixel coords
-- normalized to [0..1]. We can then normalize the coordinates of
-- a 256x256 image to the same range and feed them through the
-- network to generate an upscaled and smoothed version of the
-- original image.
-- Credit to Tsoding Daily for the idea: https://www.youtube.com/@TsodingDaily
-- ==============================================================

local network = require("network")
local instance = require("instance")
local v = require("visualizer")

-- network data
local imgPath = "networks/10525.png"
local pixels, imgw, imgh
local inputs, expected
local inputBatches, expectedBatches
local nn

-- graph data
local graphdata

-- surfaces
local sw, sh = love.graphics.getWidth(), love.graphics.getHeight()
local grid = {4, 2}

local inputSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 1))
local outputSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 2))
local graphSurface = instance.createSurface(v.gridCell(grid, sw, sh, 2, 1))
local upscaleSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 3))
local networkSurface = instance.createSurface(v.gridCell(grid, sw, sh, 2, 2))
local textSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 4))

local function drawPixels(x, y, w, h, pixelArr, scalar)
	scalar = scalar or 1
	love.graphics.push("all")
	for i=0, w - 1 do
		for j=0, h - 1 do
			love.graphics.setColor(pixelArr[w*i + j + 1])
			love.graphics.rectangle("fill", x + i*scalar, y + j*scalar, scalar, scalar)
		end
	end
	love.graphics.pop()
end

local function load()
	pixels = {}
	expected = {}
	inputs = {}
	graphdata = {}
	nn = network:new({2, 8, 8, 3})

	local imgData = love.image.newImageData(imgPath)
	imgw, imgh = imgData:getWidth(), imgData:getHeight()
	for i=0, imgw - 1 do
		for j=0, imgh - 1 do
			local r, g, b, a = imgData:getPixel(i, j)
			expected[imgw*i + j + 1] = {r, g, b}
			pixels[imgw*i + j + 1] = {r, g, b, 1}
			table.insert(inputs, {i/imgw, j/imgh})
		end
	end

	inputBatches, expectedBatches = instance.convertToBatches(8, inputs, expected)
end

local iter = 0
local highCost = -math.huge
local function update()
	iter = iter + 1
	instance.shuffleArray(inputBatches, expectedBatches)
	for i=1, #inputBatches do
		local trainArr, expectArr = inputBatches[i], expectedBatches[i]
		nn:learn(trainArr, expectArr, 1)
	end
	instance.shuffleBatches(inputBatches, expectedBatches)
end

local inputDrawn = false
local drawUpscaledImage = false

local function draw()
	local scalar = 8

	if not inputDrawn then
		local x = inputSurface.w/2 - imgw*scalar/2
		local y = inputSurface.h/2 - imgh*scalar/2
		inputDrawn = true
		love.graphics.setCanvas(inputSurface.canvas)
			drawPixels(x, y, imgw, imgh, pixels, scalar)
	end

	if drawUpscaledImage then
		drawUpscaledImage = false
		love.graphics.setCanvas(upscaleSurface.canvas)
			local upscaledPixels = {}
			local x = upscaleSurface.w/2 - 256/2
			local y = upscaleSurface.h/2 - 256/2
			for i=0, 256 - 1 do
				for j=0, 256 - 1 do
					local val = nn:forward({i/256, j/256})
					upscaledPixels[256*i + j + 1] = {val[1], val[2], val[3], 1}
				end
			end
			drawPixels(x, y, 256, 256, upscaledPixels)
	end

	if iter % 50 == 0 then
		love.graphics.setCanvas(outputSurface.canvas)
			local pixelsOut = {}
			for i=0, imgw - 1 do
				for j = 0, imgh - 1 do
					local val = nn:forward(inputs[imgw*i + j + 1])
					pixelsOut[imgw*i + j + 1] = {val[1], val[2], val[3], 1}
				end
			end
			local x = outputSurface.w/2 - imgw*scalar/2
			local y = outputSurface.h/2 - imgh*scalar/2
			drawPixels(x, y, imgw, imgh, pixelsOut, scalar)
	end

	if iter % 10 == 0 then
		local lastCost = nn:cost(inputs, expected)
		table.insert(graphdata, lastCost)
		if lastCost > highCost then highCost = lastCost end

		love.graphics.setCanvas(graphSurface.canvas)
			v.drawGraph(graphSurface, graphdata, 0, highCost)
	end

	love.graphics.setCanvas(networkSurface.canvas)
		v.drawNetwork(networkSurface, nn, false)

	love.graphics.setCanvas(textSurface.canvas)
		v.drawText(textSurface, "The leftmost view shows the source image, the next view shows the network's " ..
		"current picture of the source. Press the spacebar to see an upscaled version of the network's picture " ..
		"of the low resolution image, displayed in the third space")
	love.graphics.setCanvas()
	instance.renderSurface(inputSurface)
	instance.renderSurface(outputSurface)
	instance.renderSurface(graphSurface)
	instance.renderSurface(upscaleSurface)
	instance.renderSurface(networkSurface)
	instance.renderSurface(textSurface)

	v.renderGrid(grid, sw, sh)
end

local function keypressed(key, scancode, isrepeat)
	if key == "space" then
		drawUpscaledImage = true
	end
end

return {
	label = "Image upscaler",
	load = load,
	update = update,
	draw = draw,
	keypressed = keypressed,
}