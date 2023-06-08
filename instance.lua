-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- A collection of helper functions for instances
-- ==============================================================

local function createSurface(x, y, w, h)
	local canvas = love.graphics.newCanvas(w, h)
	return {
		canvas = canvas,
		x = x,
		y = y,
		w = w,
		h = h
	}
end

local function renderSurface(surface)
	love.graphics.draw(surface.canvas, math.floor(surface.x), math.floor(surface.y))
end

local function convertToBatches(batchCount, data1, data2)
	assert(type(batchCount) == "number" and batchCount > 1, "batch count must be a number greater than 1")
	assert(data1, "no data provideded")
	if data2 then assert(#data1 == #data2, "can only convert two arrays of equal length") end

	local step = math.floor(#data1/batchCount)
	local batches1 = {}
	local batches2 = {}
	for i=1, #data1, step do
		if #data1 - i <= step then
			table.insert(batches1, {unpack(data1, i, #data1)})
			if data2 then
				table.insert(batches2, {unpack(data2, i, #data2)})
			end
		else
			table.insert(batches1, {unpack(data1, i, i + step)})
			if data2 then
				table.insert(batches2, {unpack(data2, i, i + step)})
			end
		end
	end

	if data2 then
		return batches1, batches2
	else
		return batches1
	end
end

local function shuffleArray(data1, data2)
	assert(data1, "no data provideded")
	if data2 then assert(#data1 == #data2, "can only shuffle two arrays of equal length") end

	for i=1, #data1 do
		local j = math.random(i, #data1)
		data1[i], data1[j] = data1[j], data1[i]
		if data2 then
			data2[i], data2[j] = data2[j], data2[i]
		end
	end
end

local function shuffleBatches(data1, data2)
	assert(data1, "no data provideded")
	if data2 then assert(#data1 == #data2, "can only shuffle two arrays of equal length") end

	for i=1, #data1 do
		if data2 then
			shuffleArray(data1[i], data2[i])
		else
			shuffleArray(data1[i])
		end
	end
end

return {
	createSurface = createSurface,
	renderSurface = renderSurface,
	convertToBatches = convertToBatches,
	shuffleArray = shuffleArray,
	shuffleBatches = shuffleBatches,
}