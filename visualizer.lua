-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- A visulization library. 
-- ==============================================================

local rad = 10
local linew = 1
local posBCol = { love.math.colorFromBytes(0x00, 0x00, 0xFF, 0xFF) }
local negBCol = { love.math.colorFromBytes(0x05, 0x05, 0x20, 0xFF) }
local posWCol = { love.math.colorFromBytes(0x00, 0xFF, 0x00, 0xFF) }
local negWCol = { love.math.colorFromBytes(0xFF, 0x00, 0xFF, 0xFF) }

love.graphics.setNewFont(14)

local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

local function squash(val, min, max, newMin, newMax)
	local norm = (val - min)/(max - min)
	return norm*(newMax - newMin) + newMin
end

local function blendColorChannel(c1, c2, t)
    return (1 - t)*c1 + t*c2
end

local function blendColors(c1, c2, t)
	local blended = {}
		blended[1] = blendColorChannel(c1[1], c2[1], t)
		blended[2] = blendColorChannel(c1[2], c2[2], t)
		blended[3] = blendColorChannel(c1[3], c2[3], t)
		blended[4] = blendColorChannel(c1[4], c2[4], t)
	return blended
end

--[[ 
	draws a grid of dimensions wXh and subdivides it based on the rows table.
	each element of the rows table can be a number or another table. if it is a number N,
	that row is evenly divided into N columns. if it is a table with M elements, the row
	is divided into M columns, where the width of each column is a fraction of the sum of
	the elements. for instance, a row like {3, 1} will be divided into two columns, where
	the first column takes up 3/4ths of the row and the second column takes up the remaining
	1/4th.
	for example, a rows array like:
	rows = {
		3,
		{1, 2, 3},
		2,
	}
	will create a grid like:
	_____________
	|___|___|___|
	|__|___|____|
	|_____|_____|
]]
local function renderGrid(rows, w, h, lw)
	local ystep = h/#rows
	lw = lw or 1
	love.graphics.setLineWidth(lw)

	for i, cols in ipairs(rows) do
		-- horizontal line
		love.graphics.line(0, ystep*i, w, ystep*i)

		-- vertical lines
		if type(cols) == "number" then
			local xstep = w/cols
			for j=0, cols do
				love.graphics.line(xstep*j, ystep*(i - 1), xstep*(j), ystep*(i))
			end
		elseif type(cols) == "table" then
			local total = 0
			for j=1, #cols do
				assert(type(cols[j]) == "number", "invalid grid format, column elements must be of type number")
				total = total + cols[j]
			end
			local xstep = w/total
			local steps = 0
			for j=1, #cols + 1 do
				love.graphics.line(xstep*steps, ystep*(i - 1), xstep*steps, ystep*(i))
				if j < #cols + 1 then
					steps = steps + cols[j]
				else
					steps = steps + 1
				end
			end
		else
			error("invalid grid format, row elements must be of type number or table")
		end
	end
end

-- returns the x position, y position, width, and height of any cell created by renderGrid()
local function gridCell(rows, w, h, row, col, gap)
	gap = gap or 0
	local cols = rows[row]
	local ystep = h/#rows
	local x, xstep, width
	if type(rows[row]) == "number" then
		xstep = w/rows[row]
		x = xstep*(col - 1) + gap
		width = xstep - gap*2
	elseif type(rows[row]) == "table" then
		local total = 0
		local steps = 0
		for j=1, #cols do
			assert(type(cols[j] == "number"), "invalid grid format, column elements must be of type number")
			total = total + cols[j]
			if j < col then
				steps = steps + cols[j]
			end
		end
		xstep = w/total
		x = xstep*steps + gap
		width = xstep*cols[col] - gap*2
	else
		error("invalid grid format, row elements must be of type number or table")
	end

	local y = ystep*(row - 1) + gap
	local height = ystep - gap*2

	return x, y, width, height
end

-- visulizes the network. by default, the inputs to the network aren't drawn because
-- some networks have a comparatively very large number of input neurons compared
-- to the numbers of neurons in the hidden layers. drawing the weights of e.g.
-- the MNIST digit network inputs results in over 7000 lines between them and the
-- first hidden layer, which is impossible to make sense of visually and slow to draw
local function drawNetwork(surface, N, drawInputs)
	local w, h = surface.w, surface.h
	local sub = drawInputs and 0 or 1
	local xstep = (w/(#N.layers + 1 - sub))
	local ystep

	love.graphics.clear()
	-- draw connections
	for i=1, #N.layers do
		local l = N.layers[i]
		local x = xstep*(i + 1 - sub) - xstep/2
		ystep = (h/l.numNodes)
		for j=1, l.numNodes do
			local y = ystep*j - ystep/2
			local px = x - xstep
			local pystep
			if i > 1 then
				pystep = (h/N.layers[i - 1].numNodes)
			else
				pystep = (h/N.layers[1].numInputs)
			end
			if (i == 1 and drawInputs) or i > 1 then
				for k=1, l.numInputs do
					local py = pystep*k - pystep/2
					local weight = N.layers[i].weights[j][k]
					local lw = linew + math.abs(weight)
					local color = blendColors(negWCol, posWCol, sigmoid(weight))
					love.graphics.setColor(color)
					love.graphics.setLineWidth(lw)
					love.graphics.line(x, y, px, py)
				end
			end
		end
	end

	-- draw neurons
	for i=1, #N.layers do
		local l = N.layers[i]
		local x = xstep*(i + 1 - sub) - xstep/2
		ystep = (h/l.numNodes)
		for j=1, l.numNodes do
			local y = ystep*j - ystep/2
			local b = N.layers[i].biases[j]
			local color = blendColors(negBCol, posBCol, sigmoid(b))
			love.graphics.setColor(color)
			love.graphics.circle("fill", x, y, rad + math.abs(b))
			love.graphics.setColor(1, 1, 1, 1)
			love.graphics.setLineWidth(1)
			love.graphics.circle("line", x, y, rad + math.abs(b))
		end
	end

	-- draw inputs
	if drawInputs then
		for i=1, N.layers[1].numInputs do
			ystep = (h/N.layers[1].numInputs)
			love.graphics.setColor(0.2, 0.2, 0.8, 1)
			love.graphics.circle("fill", xstep/2, ystep*i - ystep/2, rad)
			love.graphics.setColor(1, 1, 1, 1)
			love.graphics.setLineWidth(1)
			love.graphics.circle("line", xstep/2, ystep*i - ystep/2, rad)
		end
	end
	love.graphics.setLineWidth(1)
	love.graphics.setColor(1, 1, 1, 1)
end

local function drawLayerHeatmap(x, y, w, h, weights, biases)
	local innerGap = 2

	local bmax, bmin = -math.huge, math.huge
	for _, b in ipairs(biases) do
		if b > bmax then bmax = b end
		if b < bmin then bmin = b end
	end

	for i, nodeWeights in ipairs(weights) do
		local b = squash(biases[i], bmin, bmax, 0, 1)
		local bfactor = (1 - b)*10
		for j, weight in ipairs(nodeWeights) do
			local color = blendColors(negWCol, posWCol, sigmoid(weight))
			love.graphics.setColor(color)
			love.graphics.rectangle("fill",
				x + innerGap + w/#weights*(i - 1) + bfactor,
				y + innerGap + h/#nodeWeights*(j - 1),
				w/#weights - innerGap*2 - bfactor*2,
				h/#nodeWeights - innerGap*2)
		end
	end
end

local function drawLayerActivationHeatmap(x, y, w, h, weightArr)
	local innerGap = 2
	for i, nodeWeights in ipairs(weightArr) do
		for j, weight in ipairs(nodeWeights) do
			local color = blendColors({0.01, 0.01, 0.01, 1}, {0.01, 1, 0.01, 1}, math.max(0.05, math.min(1, weight)))
			love.graphics.setColor(color)
			love.graphics.rectangle("fill",
				x + innerGap + w/#nodeWeights*(j - 1),
				y + innerGap + h/#weightArr*(i - 1),
				w/#nodeWeights - innerGap*2,
				h/#weightArr - innerGap*2)
		end
	end
end

local function drawActivationHeatmap(surface, N)
	love.graphics.clear()

	local layerTable = {}
	table.insert(layerTable, {N.layers[1].inputs})

	local grid = {1}
	for l=1, #N.layers do
		table.insert(grid, 1)
		table.insert(layerTable, {N.layers[l].outputs})
	end

	for i=1, #layerTable do
		local x, y, w, h = gridCell(grid, surface.w, surface.h, i, 1, 10)
		drawLayerActivationHeatmap(x, y, w, h, layerTable[i])
	end
	love.graphics.setColor(1, 1, 1, 1)
end

local function drawNetworkHeatmap(surface, N)
	love.graphics.clear()

	 local grid = {}
	 for _=1, #N.layers do
		table.insert(grid, 1)
	 end

	 for l=1, #N.layers do
		local x, y, w, h = gridCell(grid, surface.w, surface.h, l, 1, 10)
		drawLayerHeatmap(x, y, w, h, N.layers[l].weights, N.layers[l].biases)
	end
	love.graphics.setColor(1, 1, 1, 1)
end

-- visualizes the outputs of the provided network and highlights the highest neuron
-- activation
local function drawOutputs(surface, N)
	love.graphics.clear()
	local w, h = surface.w, surface.h
	local r = 20
	local ystep = (h/N.layers[#N.layers].numNodes)
	local highIdx, highVal, highPx, highPy = -math.huge, -math.huge, 0, 0
	local th = love.graphics.getFont():getHeight()
	local tw = love.graphics.getFont():getWidth("o1")

	for i=1, N.layers[#N.layers].numNodes do
		local x = w/3
		local y = ystep*i - ystep/2
		local val = N.layers[#N.layers].outputs[i]
		if val > highVal then
			highIdx, highVal, highPx, highPy = i, val, x, y
		end
		love.graphics.setColor(0.1, 0.1, 0.6, 1)
		love.graphics.circle("fill", x, y, r)
		love.graphics.setColor(1, 1, 1, 1)
		love.graphics.setLineWidth(1)
		love.graphics.circle("line", x, y, r)
		love.graphics.print("o" .. tostring(i - 1), x - tw/2, y - th/2)
		love.graphics.print(string.format(": %.2f", val), x + r + 4, y - th/2)
	end
	love.graphics.setColor(0, 0, 1, 1)
	love.graphics.circle("fill", highPx, highPy, r)
	love.graphics.setColor(1, 1, 1, 1)
	love.graphics.setLineWidth(3)
	love.graphics.circle("line", highPx, highPy, r)
	love.graphics.print("o" .. tostring(highIdx - 1), highPx - tw/2, highPy - th/2)
	love.graphics.print(string.format(": %.2f", highVal), highPx + r + 4, highPy - th/2)
end

local function drawText(surface, str)
	love.graphics.clear()
	love.graphics.printf(str, 2, 0, surface.w - 2, "left")
end

local function drawGraph(surface, data, min, max, color)
	if #data == 0 then return end
	love.graphics.clear()
	love.graphics.push("all")
	love.graphics.setLineWidth(1)
	color = color or {1, 0, 0, 1}
	love.graphics.setColor(color)

	local xstep = 1
	local w, h = surface.w, surface.h

	if #data > w then
		xstep = w/#data
	end

	local norm = (data[1] - min) / (max - min)
	local px = xstep
	local py = h - norm*h
	for i=2, #data do
		norm = (data[i] - min) / (max - min)
		local x = xstep*i
		local y = h - norm*h
		love.graphics.line(px, py, x, y)
		px = x
		py = y
	end

	love.graphics.pop()
end

return {
	drawNetwork = drawNetwork,
	drawOutputs = drawOutputs,
	renderGrid = renderGrid,
	gridCell = gridCell,
	drawGraph = drawGraph,
	drawText = drawText,
	drawNetworkHeatmap = drawNetworkHeatmap,
	drawActivationHeatmap = drawActivationHeatmap,
}