--[[
   Copyright (c) 2016-present, Facebook, Inc.
   All rights reserved.

   This source code is licensed under the BSD-style license found in the
   LICENSE file in the root directory of this source tree. An additional grant
   of patent rights can be found in the PATENTS file in the same directory.
]]--

local tnt = require 'torchnet.env'
local tds = require 'tds'
local argcheck = require 'argcheck'
local transform = require 'torchnet.transform'

local ImageDataset, Dataset = torch.class('tnt.ImageDataset', 'tnt.Dataset', tnt)

local function loadImage(filename)
  local fin = torch.DiskFile(filename, 'r')
  fin:binary()
  fin:seekEnd()
  local file_size_bytes = fin:position() - 1
  fin:seek(1)
  local img_binary = torch.ByteTensor(file_size_bytes)
  fin:readByte(img_binary:storage())
  fin:close()
  local img =
      (img_binary[1] == 0xff and img_binary[2] == 0xd8) and -- In case of JPEG
      image.decompressJPG(img_binary) or
      (img_binary[1] == 0x89 and img_binary[2] == 0x50) and -- In case of PNG
      image.decompressPNG(img_binary)
  return img
end

ImageDataset.__init = argcheck{
   doc = [[
<a name="ImageDataset">
#### tnt.ImageDataset(@ARGP)
@ARGT

Returns a dataset collects all image in specified directory. The dataset covers
JPEG format and PNG format.

It automates 'listdataset', which needs image loading function explicitly.

]],
   {name='self', type='tnt.ListDataset'},
   {name='path', type='string'}
   call =
      function(self, path)
         Dataset.__init(self)
         self.path = path
         self.files = {}
         for filename in paths.iterfiles(path) do
            local ext = paths.extname(filename)
            if ext == 'jpg' or ext == 'png' then
               table.insert(self.files, paths.concat(path, filename))
            end
         end
      end
}

ListDataset.size = argcheck{
   {name='self', type='tnt.ListDataset'},
   call =
      function(self)
         return #self.files
      end
}

ListDataset.get = argcheck{
   {name='self', type='tnt.ListDataset'},
   {name='idx', type='number'},
   call =
      function(self, idx)
         assert(idx >= 1 and idx <= self:size(), 'out of bound')
         return loadImage(string.format("%s/%s", self.path, self.files[idx]))
      end
}
