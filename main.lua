-- ==============================================================
-- Neucremental, a neural network creation and visualization tool
-- Copyright (c) 2023 Devin K., MIT License
-- ==============================================================

-- instances
local dir = "networks"
local foundInstances = {}
local currentInstance
local instanceSelected = 0

-- network selector
local buttons = {}
local boxMargin = 75
local boxX, boxY = boxMargin, boxMargin
local sw, sh = love.graphics.getWidth(), love.graphics.getHeight()
local boxW = sw - boxMargin*2
local boxH = sh - boxMargin*2

local function makeButton(idx, text)
	local btnWidth, btnHeight = 2*sw/3, 50
	local innerMargin, spacing = 10, 15
	local x = boxMargin + boxW/2 - btnWidth/2
	local y = boxMargin + innerMargin + (btnHeight + spacing)*(idx - 1)

	return {idx=idx, x=x, y=y, text=text, w=btnWidth, h=btnHeight}
end

local function drawNetworkSelector()
	local borderCol = {73/255, 116/255, 230/255, 1}
	local bgCol = {44/255, 138/255, 44/255, 1}

	love.graphics.setColor(bgCol)
	love.graphics.rectangle("fill", boxX, boxY, boxW, boxH)
	love.graphics.setLineWidth(3)
	love.graphics.setColor(borderCol)
	love.graphics.rectangle("line", boxX, boxY, boxW, boxH)

	for _, btn in ipairs(buttons) do
		local tw = love.graphics.getFont():getWidth(btn.text)
		local th = love.graphics.getFont():getHeight()
		local tx = btn.x + btn.w/2 - tw/2
		local ty = btn.y + btn.h/2 - th/2
		love.graphics.setColor(bgCol[1]*0.7, bgCol[2]*0.7, bgCol[3]*0.7, 1)
		love.graphics.rectangle("fill", btn.x, btn.y, btn.w, btn.h)
		love.graphics.setColor(borderCol)
		love.graphics.rectangle("line", btn.x, btn.y, btn.w, btn.h)
		love.graphics.setColor(1, 1, 1, 1)
		love.graphics.print(btn.text, tx, ty)
	end
end

function love.load()
	local items = love.filesystem.getDirectoryItems(dir)
	local instanceNum = 0
	for _, fullname in pairs(items) do
		local ext = string.sub(fullname, #fullname - 2, #fullname)
		if ext == "lua" then
			instanceNum = instanceNum + 1

			local filename = string.sub(fullname, 1, #fullname - 4)
			local instance = require(dir .. "." .. filename)
			assert(instance.load and instance.update and instance.draw, "instances must implement load, update and draw functions")
			instance.label = instance.label or "Unlabled network"

			local button = makeButton(instanceNum, instance.label)
			table.insert(foundInstances, instance)
			table.insert(buttons, button)
		end
	end
end

function love.update(dt)
	if instanceSelected > 0 then
		currentInstance = foundInstances[instanceSelected]
		instanceSelected = 0
		currentInstance.load()
	end

	if currentInstance then
		currentInstance.update(dt)
	else
		if love.mouse.isDown(1) then
			local mx, my = love.mouse.getPosition()
			for i, btn in ipairs(buttons) do
				if mx > btn.x and mx < btn.x + btn.w and my > btn.y and my < btn.y + btn.h then
					instanceSelected = btn.idx
				end
			end
		end
	end
end

function love.draw()
	if currentInstance then
		currentInstance.draw()
	elseif instanceSelected > 0 then
		local text = "Loading instance \"" .. foundInstances[instanceSelected].label .. "\""
		local tw = love.graphics.getFont():getWidth(text)
		love.graphics.print(text, sw/2 - tw/2, sh/2)
	else
		drawNetworkSelector()
	end
end

function love.keypressed(key, scancode, isrepeat)
	if key == "escape" then
		if currentInstance then
			currentInstance = nil
		else
			love.event.push("quit", "exited normally")
		end
	elseif currentInstance and currentInstance.keypressed then
		currentInstance.keypressed(key, scancode, isrepeat)
	end
end

function love.mousepressed(x, y, button)
	if currentInstance and currentInstance.mousepressed then
		currentInstance.mousepressed(x, y, button)
	end
end