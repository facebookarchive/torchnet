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
   {name="file", type="torch.File", opt=true},
   {name="filename", type="string", opt=true},
   {name="append", type="boolean", default=false},
   call =
      function(file, filename, append)
         assert(not file or not filename, "file or filename expected (not both)")
         if filename then
            file = torch.DiskFile(filename, append and "rw" or "w")
         end
         return function(data, key, value)
            if key == '__status__' then
               local status = tostring(value)
               if file then
                  file:seekEnd()
                  file:writeString(status)
                  file:writeString("\n")
               else
                  print(status)
               end
            end
         end
      end
}

return status
