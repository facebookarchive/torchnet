local __main__ = package.loaded['torchnet.env'] == nil

local tnt = require 'torchnet.env'
local tds = require 'tds'

if __main__ then
   require 'torchnet'
end

local tester = torch.Tester()
tester:add(paths.dofile('datasets.lua')(tester))
tester:add(paths.dofile('iterators.lua')(tester))
tester:add(paths.dofile('meters.lua')(tester))

function tnt.test(tests)
   tester:run(tests)
   return tester
end

if __main__ then
   require 'torchnet'
   if #arg > 0 then
      tnt.test(arg)
   else
      tnt.test()
   end
end
