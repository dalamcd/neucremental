-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- A network which trains on the MNIST handwritten digit dataset
-- and provides a simple drawing interface to test the network
--
--  MNIST handwritten digit dataset: https://yann.lecun.com/exdb/mnist/
-- ==============================================================

local v = require("visualizer")
local network = require("network")
local instance = require("instance")

-- pixel data
local pixels

-- graph data
local graphdata
local highAvg
local totalCost
local costSamples

-- network data
local nn
local training = true
local trainingInputs
local expectedOutputs
local inputBatches, expectedBatches
local updateOutputView = false
local iterations
local drawInputs = false

-- surfaces
local sw, sh = love.graphics.getWidth(), love.graphics.getHeight()
local grid = {{2, 2, 1, 1}, {3, 2}}

local networkSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 1))
local outputSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 3))
local graphSurface = instance.createSurface(v.gridCell(grid, sw, sh, 2, 1))
local pixelSurface = instance.createSurface(v.gridCell(grid, sw, sh, 2, 2))
local textSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 4))
local cakeSurface = instance.createSurface(v.gridCell(grid, sw, sh, 1, 2))

-- helper function to read the MNIST dataset
local function read_bytes(file)
	local b4 = string.byte(file:read(1))
	local b3 = string.byte(file:read(1))
	local b2 = string.byte(file:read(1))
	local b1 = string.byte(file:read(1))
	-- convert from little-endian to big-endian (I think, maybe vice versa, I can never remember) 
	return b1 + b2*256 + b3*65536 + b4*16777216
end

-- draws the pixel view and canvas
local function drawPixelGrid(surface, gw, gh)
	love.graphics.clear()
	local w, h = surface.w, surface.h
	local cellW, cellH = w/(gw + 2), h/(gh + 2)
	local wm = cellW
	local hm = cellH

	-- draw pixels
	for row=0, 27 do
		for col=0, 27 do
			local value = pixels[28*(row) + col + 1]
			local x = wm + (col)*cellW
			local y = hm + (row)*cellH
			love.graphics.setColor(value, 0, 0, 1)
			love.graphics.rectangle("fill", x, y, cellW, cellH)
		end
	end

	-- draw grid
	love.graphics.setColor(1, 1, 1, 1)
	love.graphics.setLineWidth(1)

	-- cols
	for i=1, gh + 2 do
		love.graphics.line(wm, cellH*i, w - wm, cellH*i)
	end
	-- rows
	for i=1, gw + 2 do
		love.graphics.line(cellW*i, hm, cellW*i, h - hm)
	end
end

-- returns a table with the indices of all the neighbors of cell on a grid represented by
-- a flat array/1-dimensional matrix
local function cellNeighbors(r, c, gridwidth, gridheight)
	local neighbors = {}
	local idx = gridwidth*r + c + 1

	-- left neighbor
	if c > 0 then
		table.insert(neighbors, idx - 1)
		-- top left/bottom left neighbors
		if r > 0 then
			table.insert(neighbors, idx - gridwidth - 1)
		end
		if r < gridheight - 1 then
			table.insert(neighbors, idx + gridwidth - 1)
		end
	end

	-- right neighbor
	if c < gridwidth - 1 then
		-- top right/bottom right neighbors
		table.insert(neighbors, idx + 1)
		if r > 0 then
			table.insert(neighbors, idx - gridwidth + 1)
		end
		if r < gridheight - 1 then
			table.insert(neighbors, idx + gridwidth + 1)
		end
	end

	-- top/bottom neighbors
	if r > 0 then
		table.insert(neighbors, idx - gridwidth)
	end
	if r < gridheight - 1 then
		table.insert(neighbors, idx + gridwidth)
	end

	return neighbors
end

local function clearCanvas()
	for i=1, #pixels do
		pixels[i] = 0
	end
end

local function load()
	local images = assert(io.open("networks/t10k-images.idx3-ubyte", "rb"))
	local labels = assert(io.open("networks/t10k-labels.idx1-ubyte", "rb"))
	love.graphics.setDefaultFilter("nearest", "nearest")

	nn = network:new({784, 16, 16, 10})
	pixels = {}
	graphdata = {}
	trainingInputs = {}
	expectedOutputs = {}
	highAvg = -math.huge
	totalCost = 0
	costSamples = 0
	iterations = 0

	-- load MNIST dataset
	if images and labels then
		-- ignore magic number/image count/rows/cols
		read_bytes(images) -- magic number
		read_bytes(images) -- image count
		read_bytes(images) -- rows
		read_bytes(images) -- cols

		read_bytes(labels) -- magic number
		read_bytes(labels) -- image count

		-- read 10000 samples from the dataset
		local samples = 10000
		-- pick a random sample to pre-populate the pixel view with
		local sampleIteration = math.random(1, samples)
		for i=1, samples do
			local imgArr = {}
			local imgRaw = images:read(28*28)
			local imgData = love.image.newImageData(28, 28, "r8", imgRaw)
			local imgExpected = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
			local imgLabel = string.byte(labels:read(1))
			imgExpected[imgLabel + 1] = 1
			for r=0, 27 do
				for c=0, 27 do
					imgArr[28*r + c + 1] = imgData:getPixel(c, r)
					if i == sampleIteration then
						pixels[28*r + c + 1] = imgData:getPixel(c, r)
					end
				end
			end
			table.insert(trainingInputs, imgArr)
			table.insert(expectedOutputs, imgExpected)
		end
	end

	-- stochastic gradient descent: split the data into batches and shuffle each batch. for each
	-- iteration of training, we will use one of the batches. this will make the cost jump around
	-- for each iteration, as the network is receiving only a portion of the possible inputs, but 
	-- the cost will still on average approach its limit over time. this means we can significantly
	-- speed up training by processing only a fraction of the dataset.
	inputBatches, expectedBatches = instance.convertToBatches(500, trainingInputs, expectedOutputs)
	instance.shuffleBatches(inputBatches, expectedBatches)
end

local function update(dt)
	-- detect and handle mouse input in the pixel view
	if love.mouse.isDown(1, 2) then
		local mx, my = love.mouse.getPosition()
		if mx > pixelSurface.x and mx < pixelSurface.x + pixelSurface.w
		and my > pixelSurface.y and my < pixelSurface.y + pixelSurface.h then
			local px, py = 0, 0
			px = math.floor((mx - pixelSurface.x)/(pixelSurface.w/30))
			py = math.floor((my - pixelSurface.y)/(pixelSurface.h/30))
			local idx = 28*(py - 1) + (px - 1) + 1
			if idx > 0 and idx < #pixels then
				-- indicate that the pixel grid has changed and we should have the network
				-- make a prediction on the new set of pixels
				updateOutputView = true
				local neighbors = cellNeighbors(py - 1, px - 1, 28, 28)
				if love.mouse.isDown(1) then
					local val = 15*dt
					pixels[idx] = 1
					for _, gridIndex in ipairs(neighbors) do
						pixels[gridIndex] = math.min(pixels[gridIndex] + val, 1)
					end
				elseif love.mouse.isDown(2) then
					pixels[idx] = 0
					for _, gridIndex in ipairs(neighbors) do
						pixels[gridIndex] = 0
					end
				end
			end
		end
	end

	-- train the network
	if training then
		local idx = (iterations % #inputBatches) + 1
		if idx == 1 then
			instance.shuffleArray(inputBatches, expectedBatches)
			instance.shuffleBatches(inputBatches, expectedBatches)
		end
		local trainArr, expectArr = inputBatches[idx], expectedBatches[idx]
		nn:learn(trainArr, expectArr, 1)
		iterations = iterations + 1
	end
end

local function draw()
	-- draw the grid for our layout
	v.renderGrid(grid, love.graphics.getWidth(), love.graphics.getHeight())

	-- update surfaces as necessary

	-- update the network picture
	love.graphics.setCanvas(networkSurface.canvas)
		v.drawNetwork(networkSurface, nn, drawInputs)

	-- update the output view if something new has been drawn in the pixel view
	-- or if still training, update the output view occasionally even if nothing new has been drawn
	if ((iterations <= 1 or iterations % 100 == 0) and training) or updateOutputView then
		updateOutputView = false
		nn:forward(pixels)
		love.graphics.setCanvas(outputSurface.canvas)
			v.drawOutputs(outputSurface, nn)
	end

	-- update the graph data
	if training then
		costSamples = costSamples + 1
		local idx = math.random(#trainingInputs)
		local lastCost = nn:cost({trainingInputs[idx]}, {expectedOutputs[idx]})
		totalCost = totalCost + lastCost
		local avg = totalCost/costSamples
		table.insert(graphdata, avg)
		-- update the highest average cost value if necessary, so we can draw our graph properly
		if avg > highAvg then highAvg = avg end
	end

	-- draw the graph
	love.graphics.setCanvas(graphSurface.canvas)
		v.drawGraph(graphSurface, graphdata, 0, highAvg)

	-- draw the pixel canvas
	love.graphics.setCanvas(pixelSurface.canvas)
		drawPixelGrid(pixelSurface, 28, 28)

	love.graphics.setCanvas(cakeSurface.canvas)
		v.drawNetworkHeatmap(cakeSurface, nn)

	local controls = "Controls:\n  i - toggle drawing inputs\n  c - clear pixel canvas\n  space - toggle training\n" ..
		"  Left mouse - draw pixels\n  Right mouse - erase pixels"
	love.graphics.setCanvas(textSurface.canvas)
		v.drawText(textSurface, controls)
	-- draw the surfaces to the main canvas
	love.graphics.setCanvas()
	instance.renderSurface(networkSurface)
	instance.renderSurface(graphSurface)
	instance.renderSurface(pixelSurface)
	instance.renderSurface(outputSurface)
	instance.renderSurface(textSurface)
	instance.renderSurface(cakeSurface)
end

local function keypressed(key, scancode, isrepeat)
	if key == 'i' then
		drawInputs = not drawInputs
	end
	if key == 'c' then
		clearCanvas()
	end
	if key == 'space' then
		training = not training
	end
end

return {
	label = "MNIST hand-written digits",
	load = load,
	update = update,
	draw = draw,
	keypressed = keypressed,
}