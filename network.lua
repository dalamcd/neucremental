-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- The neural network library. 
-- ==============================================================

local lay = require("layer")

local network = {}

math.randomseed(os.time())

function network:new(layerTable)
	assert(#layerTable > 1, "network requires minimum two layers")

	local o = {}
	local layers = {}

	for i=2, #layerTable do
		local layer = lay:new(layerTable[i - 1], layerTable[i])
		table.insert(layers, layer)
	end

	o.layers = layers
	setmetatable(o, self)
	self.__index = self

	return o
end

function network:backprop(ti, to)
	assert(#ti == #to, "training data are not same length")
	assert(#ti[1] == self.layers[1].numInputs, "number of inputs differs from number of input neurons")
	assert(#to[1] == self.layers[#self.layers].numNodes, "number of outputs differs from number of output neurons")

	-- i - training row
	-- l - layer
	-- j - neuron
	-- k - weight

	-- clear gradients from previous epochs
	for l=1, #self.layers do
		for j=1, self.layers[l].numNodes do
			self.layers[l].biasGradients[j] = 0
			for k=1, self.layers[l].numInputs do
				self.layers[l].weightGradients[j][k] = 0
			end
		end
	end

	for i=1, #ti do
		self:forward(ti[i])

		-- clear partial derivatives of the outputs of the node from previous epochs
		for l=1, #self.layers do
			for j=1, self.layers[l].numNodes do
				self.layers[l].derivs[j] = 0
			end
		end

		local last = self.layers[#self.layers]
		for j=1, last.numNodes do
			last.derivs[j] = last.outputs[j] - to[i][j]
		end

		for l=#self.layers, 1, -1 do
			for j=1, self.layers[l].numNodes do
				local a = self.layers[l].outputs[j]
				local da = self.layers[l].derivs[j]
				self.layers[l].biasGradients[j] = self.layers[l].biasGradients[j] + 2*da*a*(1 - a)
				for k=1, self.layers[l].numInputs do
					local pa
					if l > 1 then
						local w = self.layers[l].weights[j][k]
						self.layers[l - 1].derivs[k] = self.layers[l - 1].derivs[k] + 2*da*a*(1 - a)*w
						pa = self.layers[l - 1].outputs[k]
					else
						pa = ti[i][k]
					end
					self.layers[l].weightGradients[j][k] = self.layers[l].weightGradients[j][k] + 2*da*a*(1 - a)*pa
				end
			end
		end
	end

	-- calculate the average of all the gradients across all training data
	for l=1, #self.layers do
		for j=1, self.layers[l].numNodes do
			self.layers[l].biasGradients[j] = self.layers[l].biasGradients[j]/#ti
			for k=1, self.layers[l].numInputs do
				self.layers[l].weightGradients[j][k] = self.layers[l].weightGradients[j][k]/#ti
			end
		end
	end
end

-- finite difference method for training the network.
-- not used, included for demonstration purposes as it is much easier to understand (and MUCH slower)
-- than backpropagation
function network:fdiff(ti, to, epsilon)
	assert(#ti == #to, "training data are not same length")
	assert(#ti[1] == self.layers[1].numInputs, "number of inputs differs from number of input neurons")
	assert(#to[1] == self.layers[#self.layers].numNodes, "number of outputs differs from number of output neurons")

	epsilon = epsilon or 0.1
	local saved
	local c = self:cost(ti, to)
	for _, layer in ipairs(self.layers) do
		for i=1, layer.numNodes do
			for j=1, layer.numInputs do
				saved = layer.weights[i][j]
				layer.weights[i][j] = layer.weights[i][j] + epsilon
				layer.weightGradients[i][j] = (self:cost(ti, to) - c)/epsilon
				-- use the saved original value rather than subtracting epsilon, to prevent floating
				-- point errors from building up
				layer.weights[i][j] = saved
			end
			saved = layer.biases[i]
			layer.biases[i] = layer.biases[i] + epsilon
			layer.biasGradients[i] = (self:cost(ti, to) - c)/epsilon
			layer.biases[i] = saved
		end
	end
end

function network:applyGradients(rate)
	rate = rate or 1
	for _, layer in ipairs(self.layers) do
		layer:applyGradients(rate)
	end
end

function network:cost(ti, to)
	assert(#ti == #to, "training data are not same length")
	assert(#ti[1] == self.layers[1].numInputs, "number of inputs differs from number of input neurons")
	assert(#to[1] == self.layers[#self.layers].numNodes, "number of outputs differs from number of output neurons")

	local totalCost = 0
	for i=1, #ti do
		local results = self:forward(ti[i])
		for j=1, #results do
			local c = results[j] - to[i][j]
			totalCost = totalCost + c*c
		end
	end
	return totalCost/#ti
end

function network:forward(input)
	for _, layer in ipairs(self.layers) do
		input = layer:forward(input)
	end
	return input
end

function network:learn(ti, to, rate)
	rate = rate or 1
	self:backprop(ti, to)
	self:applyGradients(rate)
end

function network:print()
	for i=1, #self.layers do
		self.layers[i]:print(i)
	end
end

function network:print_gradients()
	for i=1, #self.layers do
		self.layers[i]:print_gradients(i)
	end
end

function network:print_test(ti, to)
	assert(#ti == #to, "training data are not same length")
	assert(#ti[1] == self.layers[1].numInputs, "number of inputs differs from number of input neurons")
	assert(#to[1] == self.layers[#self.layers].numNodes, "number of outputs differs from number of output neurons")

	local keyStr = "|"
	local fmtStr = "|"
	for i=1, #ti[1] do
		keyStr = keyStr .. " x" .. tostring(i) ..  " |"
		fmtStr = fmtStr .. "%.2f|"
	end
	for i=1, #to[1] do
		keyStr = keyStr .. " e" .. tostring(i) ..  " |"
		fmtStr = fmtStr .. "%.2f|"
	end
	for i=1, #to[1] do
		keyStr = keyStr .. " o" .. tostring(i) ..  " |"
		fmtStr = fmtStr .. "%.2f|"
	end

	print(keyStr)
	
	for i, _ in ipairs(ti) do
		local tmp = {}
		local result = self:forward(ti[i])
		for j=1, #ti[i] do
			table.insert(tmp, ti[i][j])
		end
		for j=1, #to[i] do
			table.insert(tmp, to[i][j])
		end
		for j=1, #result do
			table.insert(tmp, result[j])
		end

		print(string.format(fmtStr, unpack(tmp)))
	end
end

return network