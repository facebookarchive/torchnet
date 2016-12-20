--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local argcheck = require 'argcheck'

local text = argcheck{
   noordered=true,
   {name="file", type="torch.File", opt=true},
   {name="filename", type="string", opt=true},
   {name="keys", type="table"},
   {name="format", type="table", opt=true},
   {name="separator", type="string", default=" | "},
   {name="append", type="boolean", default=false},
   call =
      function(file, filename, keys__, format__, separator, append)
         assert(not file or not filename, "file or filename expected (not both)")
         if filename then
            file = torch.DiskFile(filename, append and "rw" or "w")
         end
         local keys = {}
         for idx, key in ipairs(keys__) do
            local format = format__ and format__[idx]
            if not format then
               table.insert(keys, {name=key, format=function(value) return string.format("%s %s", key, value) end})
            elseif type(format) == 'function' then
               table.insert(keys, {name=key, format=format})
            elseif type(format) == 'string' then
               table.insert(keys, {name=key, format=function(value) return string.format(format, value) end})
            else
               error('format must be a string or a function')
            end
         end
         return function(log)
            local txt = {}
            for _, key in ipairs(keys) do
               if log:get(key.name) then
                  local format = key.format(log:get(key.name))
                  assert(type(format) == 'string', string.format("value for key %s cannot be converted to string", key))
                  table.insert(txt, format)
               end
            end
            txt = table.concat(txt, separator)
            if file then
               file:seekEnd()
               file:writeString(txt)
               file:writeString("\n")
            else
               print(txt)
            end
         end
      end
}

return text
