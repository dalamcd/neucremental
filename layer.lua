-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
--
-- Represents a layer in the neural network.
-- ==============================================================

local layer = {}

local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

local function sigmoidPrime(x)
	return sigmoid(x)*(1 - sigmoid(x))
end

local function relu(x)
	return x > 0 and x or 0
end

local function reluPrime(x)
	return x > 0 and 1 or 0
end

local function leakyRelu(x)
	return x > 0 and x or x*0.01
end

local function leakyReluPrime(x)
	return x >= 0 and 1 or 0.01
end

function layer:new(numInputs, numNodes, activationFnc, activationPrime)

	local o = {}
	local weights = {}
	local biases = {}
	local weightGradients = {}
	local biasGradients = {}
	local outputs = {}

	activationFnc = activationFnc or sigmoid
	activationPrime = activationPrime or sigmoidPrime

	for i=1, numNodes do
		weights[i] = {}
		weightGradients[i] = {}
		biases[i] = math.random() - 0.5
		biasGradients[i] = 0
		outputs[i] = 0
		for j=1, numInputs do
			weights[i][j] = math.random() - 0.5
			weightGradients[i][j] = 0
		end
	end

	o.weights = weights
	o.biases = biases
	o.weightGradients = weightGradients
	o.biasGradients = biasGradients
	o.activationFnc = activationFnc
	o.activationPrime = activationPrime
	o.outputs = outputs
	o.derivs = {}
	o.weightedInputs = {}
	o.inputs = {}

	o.numNodes = numNodes
	o.numInputs = numInputs

	setmetatable(o, self)
	self.__index = self
	return o
end

function layer:applyGradients(rate)
	for i=1, self.numNodes do
		self.biases[i] = self.biases[i] - self.biasGradients[i]*rate
		for j=1, self.numInputs do
			self.weights[i][j] = self.weights[i][j] - self.weightGradients[i][j]*rate
		end
	end
end

function layer:forward(inputArr)
	local outputs = {}
	local inputs = {}
	for i=1, self.numNodes do
		outputs[i] = 0
		for j=1, self.numInputs do
			local weightedInput = (inputArr[j]*self.weights[i][j])
			outputs[i] = outputs[i] + weightedInput
		end
		inputs[i] = outputs[i] + self.biases[i]
		outputs[i] = self.activationFnc(outputs[i] + self.biases[i])
	end
	self.outputs = outputs
	self.inputs = inputArr
	self.weightedInputs = inputs
	return outputs
end

function layer:print(n)
	io.write(string.format("layer %i:\n", n))
	for i=1, self.numNodes do
		io.write("    ")
		for j=1, self.numInputs do
			io.write(string.format(" w%i%i: %.4f,", i, j, self.weights[i][j]))
		end
		io.write(string.format(" b%i: %.4f\n", i, self.biases[i]))
	end
end

function layer:print_gradients(n)
	io.write(string.format("layer %i (gradients):\n", n))
	for i=1, self.numNodes do
		io.write("    ")
		for j=1, self.numInputs do
			io.write(string.format(" w%i%i: %.4f,", i, j, self.weightGradients[i][j]))
		end
		io.write(string.format(" b%i: %.4f\n", i, self.biasGradients[i]))
	end
end

return layer