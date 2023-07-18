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
local imgPath = "networks/10057.png" -- see the networks folder for other example images, or use any 28x28 image
local pixels, imgw, imgh
local inputs, expected
local inputBatches, expectedBatches
local nn

local view = 0

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

local function screenCoordToGridCoord(mx, my, x, y, w, h, scalar)
	local lx, ly = mx - x, my - y

	if lx >= 0 and ly >= 0 then
		local gx = math.floor(lx/scalar)
		local gy = math.floor(ly/scalar)
		if gx <= w and gy <= h then
			return gx, gy
		else
			return -1, -1
		end
	else
		return -1, -1
	end

end

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
	nn = network:new({2, 7, 7, 3})

	local imgData = love.image.newImageData(imgPath)
	imgw, imgh = imgData:getWidth(), imgData:getHeight()
	for i=0, imgw - 1 do
		for j=0, imgh - 1 do
			local r, g, b, a = imgData:getPixel(i, j)
			expected[imgw*i + j + 1] = {r, g, b}
			pixels[imgw*i + j + 1] = {r, g, b, a}
			table.insert(inputs, {i/imgw, j/imgh})
		end
	end

	inputBatches, expectedBatches = instance.convertToBatches(12, inputs, expected)
end

local iterations = 0
local highCost = -math.huge
local function update()
	iterations = iterations + 1
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

	if iterations % 10 == 0 then
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

	if iterations % 50 == 0 then
		local lastCost = nn:cost(inputs, expected)
		table.insert(graphdata, lastCost)
		if lastCost > highCost then highCost = lastCost end

		love.graphics.setCanvas(graphSurface.canvas)
			v.drawGraph(graphSurface, graphdata, 0, highCost)
	end

	love.graphics.setCanvas(networkSurface.canvas)
		if view == 0 then
			v.drawNetwork(networkSurface, nn, true)
			networkSurface.forceDrawHeatmap = true
		elseif view == 1 then
			v.drawNetworkHeatmap(networkSurface, nn)
			networkSurface.forceDrawHeatmap = true
		elseif view == 2 then
			local mx, my = love.mouse.getPosition()
			local x = outputSurface.w/2 - imgw*scalar/2
			local y = outputSurface.h/2 - imgh*scalar/2
			local gx, gy = screenCoordToGridCoord(mx, my, x, y, imgw, imgh, scalar)
			if gx >= 0 then
				nn:forward({gx/imgw, gy/imgh})
				v.drawActivationHeatmap(networkSurface, nn)
			elseif networkSurface.forceDrawHeatmap then
				nn:forward({0, 0})
				v.drawActivationHeatmap(networkSurface, nn)
			end
			networkSurface.forceDrawHeatmap = false
		end

	love.graphics.setCanvas(textSurface.canvas)
		v.drawText(textSurface, "The leftmost view shows the source image, the next view shows the network's " ..
		"current picture of the source. Press the spacebar to see an upscaled version of the network's picture " ..
		"of the low resolution image displayed in the third space. Press tab to switch between network view; heatmap view; and " ..
		"an activation heatmap (mouse over the left image to see how each neuron responds to those x,y coordinates)")
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
	if key == "tab" then
		view = (view + 1) % 3
	end
end

return {
	label = "Image upscaler",
	load = load,
	update = update,
	draw = draw,
	keypressed = keypressed,
}