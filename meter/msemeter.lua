local tnt = require 'torchnet.env'
local argcheck = require 'argcheck'

local MSEMeter = torch.class('tnt.MSEMeter', 'tnt.Meter', tnt)

MSEMeter.__init = argcheck{
   {name = 'self', type = 'tnt.MSEMeter'},
   {name = 'root', type = 'boolean', default = false},
   call = function(self, root)
      self:reset()
      self.root = root
   end
}

MSEMeter.reset = argcheck{
   {name = 'self', type = 'tnt.MSEMeter'},
   call = function(self)
      self.n = 0
      self.sesum = 0
   end
}

MSEMeter.add = argcheck{
   {name = 'self',   type = 'tnt.MSEMeter'},
   {name = 'output', type = 'torch.*Tensor'},
   {name = 'target', type = 'torch.*Tensor'},
   call = function(self, output, target)
      assert(output:isSameSizeAs(target), 'output and target not the same size')
      assert(torch.isTypeOf(output, torch.typename(target)),
         'output and target not the same type')
      self.n = self.n + output:nElement()
      self.sesum = self.sesum + torch.add(output, -target):pow(2):sum()
   end
}

MSEMeter.value = argcheck{
   {name = 'self', type = 'tnt.MSEMeter'},
   call = function(self)
      local mse = self.sesum / math.max(1, self.n)
      return self.root and math.sqrt(mse) or mse
   end
}
