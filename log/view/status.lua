--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local argcheck = require 'argcheck'

local status = argcheck{
   noordered=true,
   {name="filename", type="string", opt=true},
   {name="append", type="boolean", default=false},
   call =
      function(filename, append)
         if filename and not append then
            local f = io.open(filename, 'w') -- reset the file
            assert(f, string.format("could not open file <%s> for writing", filename))
            f:close()
         end
         return function(data, key, value)
            if key == '__status__' then
               local status = tostring(value)
               if filename then
                  local f = io.open(filename, 'a+') -- append
                  assert(f, string.format("could not open file <%s> for writing", filename))
                  f:write(status)
                  f:write("\n")
                  f:close()
               else
                  print(status)
               end
            end
         end
      end
}

return status
